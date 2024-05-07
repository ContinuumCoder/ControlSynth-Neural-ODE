
import torch

class DynamicODESolver:
    def __init__(self, func, y0, gfunc=None, ufunc=None, u0=None, step_size=None, interp="linear", atol=1e-6, norm=None):
        self.func = func
        self.y0 = y0
        self.gfunc = gfunc
        self.ufunc = ufunc
        self.u0 = u0 if u0 is not None else y0
        self.step_size = step_size
        self.interp = interp
        self.atol = atol
        self.norm = norm

    def _before_integrate(self, t):
        pass

    def _advance(self, next_t):
        t0 = self.t
        y0 = self.y
        u0 = self.u
        dt = next_t - t0

        

        if self.ufunc is None:
            u1 = u0
            udot = u1
        else:
            udot = self.ufunc(t0, u0)
            u1 = udot * dt + u0


        if self.gfunc is None:
            gu1 = udot
        else:
            gu1 = self.gfunc(t0, udot)

        dy, f0 = self._step_func(t0, dt, next_t, y0, gu1)
        y1 = y0 + dy

        

        if self.interp == "linear":
            y_next = self._linear_interp(t0, next_t, y0, y1, next_t)
            u_next = self._linear_interp(t0, next_t, u0, u1, next_t)
        elif self.interp == "cubic":
            f1 = self.func(next_t, y1) + self.gfunc(next_t, self.gu)
            y_next = self._cubic_hermite_interp(t0, y0, f0, next_t, y1, f1, next_t)
        else:
            y_next = y1
            u_next = u1

        self.t = next_t
        self.y = y_next
        self.u = u_next
        return y_next

    def integrate(self, t):
        if self.step_size is None:
            self.step_size = t[1] - t[0]

        self.t = t[0]
        self.y = self.y0
        self.u = self.u0

        solution = torch.empty(len(t), *self.y0.shape, dtype=self.y0.dtype, device=self.y0.device)
        solution[0] = self.y0
        t = t.to(self.y0.device, self.y0.dtype)
        self._before_integrate(t)
        for i in range(1, len(t)):
            solution[i] = self._advance(t[i])
        return solution

    def _step_func(self, t0, dt, t1, y0, gu1):
        f0 = self.func(t0, y0) + gu1
        dy = f0 * dt 
        return dy, f0

    def _cubic_hermite_interp(self, t0, y0, f0, t1, y1, f1, t):
        h = (t - t0) / (t1 - t0)
        h00 = (1 + 2 * h) * (1 - h) * (1 - h)
        h10 = h * (1 - h) * (1 - h)
        h01 = h * h * (3 - 2 * h)
        h11 = h * h * (h - 1)
        dt = (t1 - t0)
        return h00 * y0 + h10 * dt * f0 + h01 * y1 + h11 * dt * f1

    def _linear_interp(self, t0, t1, y0, y1, t):
        if t == t0:
            return y0
        if t == t1:
            return y1
        slope = (t - t0) / (t1 - t0)
        return y0 + slope * (y1 - y0)
