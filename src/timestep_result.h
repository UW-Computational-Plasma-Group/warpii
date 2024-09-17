#pragma once

#include "timestep_request.h"

namespace warpii {
/**
 * Expresses the result of an attempt to take a timestep with a given dt.
 */
class TimestepResult {
    public:
        TimestepResult(double attempted_dt, bool successful, double achieved_dt):
            attempted_dt(attempted_dt), successful(successful), achieved_dt(achieved_dt) {}

        static TimestepResult failure(double attempted_dt) {
            return TimestepResult(attempted_dt, false, 0.0);
        }

        static TimestepResult success(const TimestepRequest& request) {
            return TimestepResult(request.requested_dt, true, request.requested_dt);
        }

        // The dt that was attempted
        double attempted_dt;
        // Whether the timestep successfully advanced by some amount, which may not
        // be the same as the dt that was initially attempted.
        bool successful;
        // If successful, the dt that was actually achieved.
        // Solution vectors have been advanced by this much.
        double achieved_dt;
};
}
