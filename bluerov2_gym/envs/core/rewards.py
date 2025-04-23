import numpy as np


class Reward:
    """
    Same class shell ─ we only touch the trajectory-tracking section
    and expose a few tunable coefficients up top.
    """

    def __init__(self):
        # ─── weights & misc ────────────────────────────────────────────────
        self.dt        = 0.1      # [s]
        self.k_cross   = 3.0      # ⊥-distance weight         ↑ (stronger pull)
        self.k_align   = 0.6      # velocity-alignment weight ↑
        self.k_prog    = 1.0      # forward-progress bonus
        self.k_head    = 0.15     # heading-error penalty     ↓ (so it doesn’t dominate)
        self.k_ctrl    = 0.01     # control-effort penalty
        self.lookahead = 5        # way-points to peek ahead
        self.finish_boost = 5.0   # one-time bonus on last wp

        # ─── running state ────────────────────────────────────────────────
        self.prev_i        = 0    # last “closest-wp” index  (for Δprogress)
        self.finish_flag   = False
        self.last_reward   = 0.0  # (optional: for debugging)

    # ──────────────────────────────────────────────────────────────────────
    # SIMPLE BASELINE REWARD  (unchanged)                                  #
    # ──────────────────────────────────────────────────────────────────────
    def get_reward(self, obs):
        position_error   = np.linalg.norm([obs["x"][0], obs["y"][0], obs["z"][0]])
        velocity_penalty = np.linalg.norm([obs["vx"][0], obs["vy"][0], obs["vz"][0]])
        orientation_err  = abs(obs["theta"][0])

        return -(
            1.0 * position_error +
            0.1 * velocity_penalty +
            0.5 * orientation_err
        )

    # ──────────────────────────────────────────────────────────────────────
    # IMPROVED TRAJECTORY REWARD                                           #
    # ──────────────────────────────────────────────────────────────────────
    def get_reward_trajectory(self, obs, action, trajectory_file):
        """
        Observation keys expected:
            x,y,z           : world-frame position  (m)
            vx,vy,vz        : world-frame velocity  (m/s)
            theta           : yaw heading           (rad)
        """
        self.reference = trajectory_file
        N_ref          = len(self.reference)

        # ── 1. nearest trajectory point ──────────────────────────────────
        p      = np.array([obs["x"][0],  obs["y"][0],  obs["z"][0]])
        dists  = np.linalg.norm(self.reference - p, axis=1)
        i_star = int(np.argmin(dists))
        p_star = self.reference[i_star]

        # cross-track error (squared → smoother gradient when close)
        e_cross = np.linalg.norm(p - p_star)
        r_cross = -self.k_cross * e_cross**2

        # ── 2. local tangent (+look-ahead) ────────────────────────────────
        i_fwd   = min(i_star + self.lookahead, N_ref - 1)
        tangent = self.reference[i_fwd] - p_star
        if np.allclose(tangent, 0.0):
            tangent = np.array([1e-6, 0, 0])  # degenerate (end of path)
        tangent /= np.linalg.norm(tangent)

        # ── 3. velocity alignment  (reward true speed in tangent dir) ────
        v          = np.array([obs["vx"][0], obs["vy"][0], obs["vz"][0]])
        v_along    = np.dot(v, tangent)          # signed
        r_align    =  self.k_align * v_along      # >0 if moving forward

        # ── 4. forward progress bonus (index-based) ──────────────────────
        prog_steps = max(0, i_star - self.prev_i)
        r_prog     = self.k_prog * prog_steps
        self.prev_i = i_star

        # ── 5. heading alignment penalty (optional) ──────────────────────
        head_err = Reward.angle_wrap(np.arctan2(tangent[1], tangent[0]) - obs["theta"][0])
        r_head   = -self.k_head * abs(head_err)

        # ── 6. control effort penalty ────────────────────────────────────
        r_ctrl = -self.k_ctrl * np.square(action).sum()

        # ── 7. terminal “finish-line” bonus ──────────────────────────────
        r_finish = 0.0
        if not self.finish_flag and i_star >= N_ref - 2:
            r_finish      = self.finish_boost
            self.finish_flag = True     # give it only once

        reward = r_cross + r_align + r_prog + r_head + r_ctrl + r_finish
        self.last_reward = reward       # handy for external logging
        return reward

    # ──────────────────────────────────────────────────────────────────────
    @staticmethod
    def angle_wrap(a):
        """ wrap to (-π, π] """
        return (a + np.pi) % (2 * np.pi) - np.pi
