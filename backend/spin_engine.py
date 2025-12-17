from .rng import RNG  # RNG wrapper for reproducible sampling
from . import config

class SpinEngine:
    def __init__(self, rng, config):
        self.rng = rng  # RNG instance (seeded upstream)
        self.config = config  # parameters such as means/stddevs and slot count

    def simulate(self):
        print("Im spinning!")
        # Random Continuous Vars: Draw angular velocity and spin duration from the configured distributions.
        omega = self.rng.normal(self.config.OMEGA_MEAN,
                                self.config.OMEGA_STD)
        tspin = self.rng.expo(self.config.TSPIN_LAMBDA)
        

        # Compute the base travel angle and add an independent phase offset.
        #Random Continuous Vars: 
        base_angle = (omega * tspin) % 360 #some random angle determined by the physical-ish part
        phase = self.rng.uniform(0.0, 360.0) #independent random “rotation” uniformly spread over [0, 360]
        friction_noise = self.rng.gamma(self.config.FRICTION_SHAPE, self.config.FRICTION_SCALE)
        #Random variable
        landing_angle = (base_angle * friction_noise + phase) % 360

        # Map the landing angle into a slot index.
        initial_slot = 360 / self.config.NUM_SLOT
        bounce_offset = self.rng.laplace(0, self.config.BOUNCE_DIVERSITY)
        final_slot = int((initial_slot + bounce_offset) % self.config.NUM_SLOT)

        # Alternatively, sample directly from the slot indices:
        # return self.rng.uniform_int(0, self.config.NUM_SLOT - 1)
        return final_slot

if __name__ == "__main__":
    from .rng import RNG
    from . import config

    engine = SpinEngine(RNG(), config)
    print(engine.simulate())
