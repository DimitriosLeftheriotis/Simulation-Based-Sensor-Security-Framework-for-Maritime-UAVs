"""
Command-and-control (C2) uplink model and Layer 1 (MAVLink 2 message signing).

Everything here runs inside the simulation loop:

* A ground control station (GCS) sends a HEARTBEAT at 1 Hz and, before
  take-off, the MISSION_START command (MAV_CMD_MISSION_START).
* Layer 1 verifies every received frame as in MAVLink 2 signing: a 48-bit
  signature (truncated SHA-256 over secret key | frame | link id | 48-bit
  timestamp) and a per-stream monotonic-timestamp replay window.  Unsigned
  frames are rejected when signing is required.
* Attack A7 on the uplink, in three realisations:
    'A7j'  jamming: a Gilbert-Elliott channel with randomised stationary loss
           and mean burst length, and a latency that grows in the bad state;
    'A7r'  replay: the attacker records the uplink and, at onset, replays the
           legitimate MISSION_START frame (which restarts the mission and pulls
           the UAV back towards the first waypoint);
    'A7i'  injection: the attacker forges SET_POSITION_TARGET frames (guided
           go-to) towards a point outside the patrol area, spoofing the GCS
           identity but without the key.
* Link-loss handling: a baseline autopilot triggers its GCS failsafe (RTL)
  after GCS_FS_TIMEOUT without a heartbeat; the framework continues the
  stored mission and escalates to CONTINGENCY after LINK_LOSS_LIMIT.

All randomness uses an RNG stream that is independent of the main simulation
RNG, so adding the link model leaves every other experiment bit-identical.
"""
import hashlib
import hmac
import struct
import binascii
import numpy as np

from .config import (DT, HB_PERIOD, GCS_FS_TIMEOUT, LINK_LOSS_LIMIT,
                     LINK_LOSS_GOOD, LINK_LOSS_BAD, LINK_LAT_GOOD, LINK_LAT_BAD,
                     A7J_PI_BAD, A7J_BURST, A7R_COPIES, A7I_PERIOD, A7I_OFFSET,
                     RECT)

MSG_HEARTBEAT = 0
MSG_COMMAND_LONG = 76
MSG_SET_POSITION_TARGET = 86
MAV_CMD_MISSION_START = 300
GCS_SYS, GCS_COMP, LINK_ID = 255, 190, 0
TS_ORIGIN = 1_000_000_000_000   # arbitrary 48-bit timestamp origin, 10 us units


def _ts(t):
    """MAVLink 2 signing timestamp (10 us units) for simulation time *t*."""
    return TS_ORIGIN + int(round((t + 10.0) * 1e5))


class Frame:
    __slots__ = ('seq', 'sysid', 'compid', 'msgid', 'payload', 'hdr', 'crc',
                 'link_id', 'ts', 'sig', 'kind')

    def __init__(self, seq, sysid, compid, msgid, payload, kind):
        self.seq, self.sysid, self.compid = seq & 0xFF, sysid, compid
        self.msgid, self.payload, self.kind = msgid, payload, kind
        self.hdr = (struct.pack('<BBBBBBB', 0xFD, len(payload), 0x01, 0x00,
                                self.seq, sysid, compid)
                    + struct.pack('<I', msgid)[:3])
        self.crc = struct.pack('<H', binascii.crc_hqx(self.hdr[1:] + payload, 0xFFFF))
        self.link_id, self.ts, self.sig = LINK_ID, 0, None


def _replay_copy(fr):
    """Byte-identical copy of a recorded frame, tagged as a replay."""
    c = Frame.__new__(Frame)
    for a in Frame.__slots__:
        setattr(c, a, getattr(fr, a))
    c.kind = 'replay'
    return c


def signature(key, fr):
    """48-bit MAVLink 2 signature: SHA-256(key | frame | link id | timestamp)[:6]."""
    return hashlib.sha256(key + fr.hdr + fr.payload + fr.crc
                          + bytes([fr.link_id]) + fr.ts.to_bytes(6, 'little')).digest()[:6]


class L1Verifier:
    """Layer 1: signature check plus monotonic-timestamp replay window."""

    def __init__(self, key):
        self.key = key
        self.last_ts = {}
        self.rejected = {'unsigned': 0, 'bad_sig': 0, 'replay': 0}

    def check(self, fr):
        if fr.sig is None:
            reason = 'unsigned'
        elif not hmac.compare_digest(signature(self.key, fr), fr.sig):
            reason = 'bad_sig'
        else:
            stream = (fr.sysid, fr.compid, fr.link_id)
            if fr.ts <= self.last_ts.get(stream, -1):
                reason = 'replay'
            else:
                self.last_ts[stream] = fr.ts
                return True, None
        self.rejected[reason] += 1
        return False, reason


class C2Link:
    """GCS uplink, channel, attacker and the autopilot's command handling."""

    def __init__(self, attacks, signing, seed, t_on, k_on, framework):
        self.rng = np.random.default_rng([int(seed), 7])
        self.key = self.rng.bytes(32)
        self.attacks = attacks
        self.signing = signing
        self.framework = framework          # link-loss policy of config F
        self.verifier = L1Verifier(self.key) if signing else None
        self.t_on = t_on
        self.k_on = k_on
        self.seq = 0
        self.queue = []                     # (deliver_k, Frame)
        # A7j channel parameters (randomised per run)
        pi_bad = self.rng.uniform(*A7J_PI_BAD)
        burst = self.rng.uniform(*A7J_BURST)
        self.p_bg = DT / burst
        self.p_gb = self.p_bg * pi_bad / (1.0 - pi_bad)
        self.bad = False
        # A7i target (direction chosen away from the patrol area) and offset
        self.inj_dist = self.rng.uniform(*A7I_OFFSET)
        self.inj_rot = np.deg2rad(self.rng.uniform(-45.0, 45.0))
        self.inj_target = None
        # autopilot command state
        self.mode = 'AUTO'
        self.k0 = 0                         # step at which the mission (re)started
        self.target = None
        self.last_hb_t = 0.0
        self.max_gap = 0.0
        # bookkeeping
        self.hijacked = False
        self.hijack_t = None
        self.restarts = 0
        self.legit_rejected = 0
        self.first_reject_t = None
        self.link_abort = False
        self.link_abort_t = None
        self.recorded = None
        self._preflight()

    # ---------------------------------------------------------------- frames
    def _gcs_frame(self, msgid, payload, t):
        fr = Frame(self.seq, GCS_SYS, GCS_COMP, msgid, payload, 'legit')
        self.seq += 1
        fr.ts = _ts(t)
        fr.sig = signature(self.key, fr)    # the GCS always signs; the
        return fr                           # receiver may or may not verify

    def _preflight(self):
        """MISSION_START sent before take-off (recorded by an A7r attacker)."""
        payload = struct.pack('<7fHBBB', 0, 0, 0, 0, 0, 0, 0,
                              MAV_CMD_MISSION_START, 1, 1, 0)
        fr = self._gcs_frame(MSG_COMMAND_LONG, payload, -1.0)
        self.recorded = fr
        self._deliver(fr, k=0, t=-1.0, preflight=True)

    def _transmit(self, fr, k, t, jammed):
        """Pass a frame through the channel; queue it for delivery if it survives."""
        loss = LINK_LOSS_BAD if (jammed and self.bad) else LINK_LOSS_GOOD
        if self.rng.random() < loss:
            return
        lat = self.rng.uniform(*(LINK_LAT_BAD if (jammed and self.bad) else LINK_LAT_GOOD))
        self.queue.append((k + max(1, int(np.ceil(lat / DT))), fr))

    def _deliver(self, fr, k, t, preflight=False):
        if self.verifier is not None:
            ok, _ = self.verifier.check(fr)
            if not ok:
                if fr.kind == 'legit':
                    self.legit_rejected += 1
                elif self.first_reject_t is None:
                    self.first_reject_t = t
                return False
        # accepted: apply it
        if fr.kind != 'legit' and not self.hijacked:
            self.hijacked, self.hijack_t = True, t
        if fr.msgid == MSG_HEARTBEAT:
            if k >= self.k_on:
                self.max_gap = max(self.max_gap, t - self.last_hb_t)
            self.last_hb_t = t
        elif fr.msgid == MSG_COMMAND_LONG:
            cmd = struct.unpack('<7fHBBB', fr.payload)[7]
            if cmd == MAV_CMD_MISSION_START:
                self.mode, self.target = 'AUTO', None
                self.k0 = 0 if preflight else k
                if not preflight:
                    self.restarts += 1
        elif fr.msgid == MSG_SET_POSITION_TARGET:
            x, y = struct.unpack('<ii', fr.payload[:8])
            self.mode, self.target = 'GUIDED', np.array([x, y], float) / 100.0
        return True

    # ------------------------------------------------------------------ step
    def step(self, k, t, uav_pos):
        """Advance the uplink by one 10 Hz step.  Returns ``link_alarm``:
        True when L1 rejected a frame at this step."""
        attacking = k >= self.k_on
        jammed = attacking and 'A7j' in self.attacks
        if jammed:
            self.bad = (self.rng.random() < self.p_gb) if not self.bad \
                else (self.rng.random() > self.p_bg)

        # legitimate GCS traffic
        if k % int(round(HB_PERIOD / DT)) == 0:
            payload = struct.pack('<IBBBBB', 0, 6, 8, 0, 4, 3)
            self._transmit(self._gcs_frame(MSG_HEARTBEAT, payload, t), k, t, jammed)

        # attacker traffic
        k_on = self.k_on
        if 'A7r' in self.attacks and attacking and k - k_on < A7R_COPIES * 5 \
                and (k - k_on) % 5 == 0:
            self._transmit(_replay_copy(self.recorded), k, t, False)   # verbatim replay
        if 'A7i' in self.attacks and attacking and \
                (k - k_on) % int(round(A7I_PERIOD / DT)) == 0:
            if self.inj_target is None:
                c = np.array([RECT / 2, RECT / 2])
                u = uav_pos - c
                u = u / max(np.linalg.norm(u), 1e-6)
                cr, sr = np.cos(self.inj_rot), np.sin(self.inj_rot)
                u = np.array([cr * u[0] - sr * u[1], sr * u[0] + cr * u[1]])
                self.inj_target = uav_pos + self.inj_dist * u
            tx, ty = (self.inj_target * 100.0).astype(int)
            payload = struct.pack('<ii', int(tx), int(ty)) + bytes(45)
            fr = Frame(self.rng.integers(256), GCS_SYS, GCS_COMP,
                       MSG_SET_POSITION_TARGET, payload, 'forged')
            fr.ts = _ts(t) + 100_000                  # newer than any legit frame
            fr.sig = self.rng.bytes(6)                # attacker has no key
            self._transmit(fr, k, t, False)

        # delivery and verification
        alarm = False
        due = [f for kk, f in self.queue if kk <= k]
        self.queue = [(kk, f) for kk, f in self.queue if kk > k]
        for fr in due:
            if not self._deliver(fr, k, t):
                alarm = True

        # link-loss handling
        gap = t - self.last_hb_t
        limit = LINK_LOSS_LIMIT if self.framework else GCS_FS_TIMEOUT
        if attacking:
            self.max_gap = max(self.max_gap, gap)
        if gap > limit and not self.link_abort:
            self.link_abort, self.link_abort_t = True, t
        return alarm

    def summary(self):
        rej = self.verifier.rejected if self.verifier else {'unsigned': 0, 'bad_sig': 0, 'replay': 0}
        return dict(hijacked=self.hijacked, hijack_t=self.hijack_t,
                    restarts=self.restarts, rejected=dict(rej),
                    legit_rejected=self.legit_rejected,
                    first_reject_t=self.first_reject_t,
                    link_abort=self.link_abort, link_abort_t=self.link_abort_t,
                    max_link_gap=self.max_gap)
