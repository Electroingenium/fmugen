"""FLORIS: the default 3-turbine farm for one wind speed and direction; returns each turbine's power (W)
and the farm power."""
from floris import FlorisModel


class Farm:
    def __init__(self):
        self.fm = FlorisModel("defaults")
        self.fm.set(layout_x=[0.0, 630.0, 1260.0], layout_y=[0.0, 0.0, 0.0])

    def run(self, wind_speed=8.0, wind_direction=270.0, turbulence_intensity=0.06):
        self.fm.set(wind_speeds=[wind_speed], wind_directions=[wind_direction],
                    turbulence_intensities=[turbulence_intensity])
        self.fm.run()
        return self.fm.get_turbine_powers()[0], float(self.fm.get_farm_power()[0])
