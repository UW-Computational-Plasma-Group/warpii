#pragma once

namespace warpii {
class TimestepRequest {
    public:
        TimestepRequest(double requested_dt, bool is_flexible):
            requested_dt(requested_dt), is_flexible(is_flexible) {}

        // The dt that is requested by the caller.
        double requested_dt;
        // Whether the requested dt is flexible. A dt may be flexible if
        // no sub-step of that size has already been taken, in which case
        // an operator is free to search for the maximum dt that it's able 
        // to take.
        bool is_flexible;
};
}
