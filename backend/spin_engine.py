from .rng import RNG  # RNG wrapper for reproducible sampling
from . import config

class SpinEngine:
    def __init__(self, rng, config):
        self.rng = rng  # RNG instance (seeded upstream)
        self.config = config  # parameters such as means/stddevs and slot count

    def simulate(self):
        # Draw angular velocity and spin duration from the configured distributions.
        omega = self.rng.normal(self.config.OMEGA_MEAN, 
                                self.config.OMEGA_STD)
        tspin = self.rng.expo(self.config.TSPIN_LAMBDA)
        base_angle = (omega * tspin) % 360
        phase = self.rng.uniform(0.0, 360.0)
        angle = (base_angle + phase) % 360
        slot_width = 360 / self.config.NUM_SLOTS
        slot = int(angle / slot_width)
        # Convert to landing angle, then map into a numbered slot.
        #slot = self.rng.uniform_int(0, self.config.NUM_SLOTS - 1)
        return {"slot": slot}

if __name__ == "__main__":
    from .rng import RNG
    from . import config

    engine = SpinEngine(RNG(), config)
    print(engine.simulate())