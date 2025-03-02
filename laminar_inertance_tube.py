import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root
from miscellaneous import first_derivative

class LaminarInertanceTube:
    def __init__(self, length:float, diameter:float, buffer_tank_volume:float = 1, absolute_roughness:float = 0):
        self.length = length
        self.diameter = diameter
        self.buffer_tank_volume = buffer_tank_volume
        self.flow_area = np.pi/4 * diameter**2
        
        self.absolute_viscosity = 1.96E-5
        self.specific_gas_constant = 2077.1
    
    def simulate(self, frequency:float, mean_pressure:float, pressure_ratio:float, mean_temperature:float, polytropic_index:float, x_divisions:int=10, t_divisions:int=1000, periods:int=10):
        # Calculating some constants
        pressure_amplitude = mean_pressure * (pressure_ratio - 1)/(pressure_ratio + 1)
        mean_density = mean_pressure/(mean_temperature * self.specific_gas_constant)
        polytropic_coefficient = (mean_pressure/mean_density**polytropic_index)
        period = 1 / frequency
        friction_coefficient = 32*self.absolute_viscosity/self.diameter**2
        tstep = period / (t_divisions)
        xstep = self.length / x_divisions
        vstep = xstep * self.flow_area
        
        # Some functions
        pulse_tube_inlet_pressure = lambda t : mean_pressure + pressure_amplitude * np.sin(2 * np.pi * frequency * t)
        pulse_tube_inlet_density = lambda t : (1/polytropic_coefficient * (mean_pressure + pressure_amplitude * np.sin(2 * np.pi * frequency * t)))**(1/polytropic_index)
        
        inlet_pressure_t_derivative = lambda t : 2 * np.pi * frequency * pressure_amplitude * np.cos(2 * np.pi * frequency * t)
        inlet_density_t_derivative = lambda t : ((2*np.pi*frequency*pressure_amplitude*np.cos(2*np.pi*frequency*t) * ((pressure_amplitude*np.sin(2*np.pi*frequency*t) + mean_pressure)/polytropic_coefficient)**(1/polytropic_index))
                                                 /(polytropic_index * (pressure_amplitude * np.sin(2*np.pi*frequency*t) + mean_pressure)))
        
        density = lambda pressure : (pressure / polytropic_coefficient)**(1/polytropic_index)
        pressure = lambda density : polytropic_coefficient * density**polytropic_index
        
        # Discretization
        velocities = np.zeros(x_divisions, dtype=float)
        pressures = np.full(x_divisions, mean_pressure, dtype=float)
        densities = np.full(x_divisions, mean_density, dtype=float)
        midpoint_densities = np.full(x_divisions - 1, mean_density, dtype=float)
        x_positions = np.linspace(0, self.length, x_divisions, dtype=float)
        times = np.linspace(0, period * periods, t_divisions*periods, dtype=float, endpoint=True)
        phase_angles_deg = np.linspace(0, 360, t_divisions, dtype=float, endpoint=True)
        buffer_tank_pressure = mean_pressure
        buffer_tank_density = mean_density
        
        pulse_tube_inlet_mass_flowrate = np.zeros(periods * t_divisions, dtype=float)
            
        # Iteration
        i = 0
        for t in times:
            
            # first x derivative arrays
            velocity_x_derivative = first_derivative(velocities, xstep)
            density_x_derivative = first_derivative(densities, xstep)
            pressure_x_derivative = first_derivative(pressures, xstep)
            
            # first t derivative arrays
            velocity_t_derivative = -1/densities*(pressure_x_derivative + friction_coefficient * velocities)
            density_t_derivative = -(densities*velocity_x_derivative + velocities*density_x_derivative)
            
            # Boundary conditions
            density_t_derivative[0] = inlet_density_t_derivative(t)
            density_t_derivative[-1] = 0
            pressure_t_derivative = polytropic_coefficient*polytropic_index*densities**(polytropic_index - 1)*density_t_derivative
            # Boundary conditions
            pressure_t_derivative[0] = inlet_pressure_t_derivative(t)
            pressure_t_derivative[-1] = 0
            
            # xt derivative arrays
            pressure_xt_derivative = first_derivative(pressure_t_derivative, xstep)
            
            # # second t derivative arrays
            velocity_tt_derivative = (
                                        densities**-2 * density_t_derivative*pressure_x_derivative
                                    - densities**-1 * pressure_xt_derivative
                                    - friction_coefficient*velocity_t_derivative
                                    )
            
            # Calculating new densities, excluding boundaries
            velocities += velocity_t_derivative * tstep + 0.5 * velocity_tt_derivative * tstep **2
            mass_flows = (velocities * densities * self.flow_area) * tstep
            midpoint_densities += (mass_flows - np.roll(mass_flows, -1))[:-1]/vstep

            # Calculaing new densities from midpoints
            #midpoint_density_gradient = first_derivative(midpoint_densities, xstep)
            #densities[1 : -1] = midpoint_densities[:-1] + midpoint_density_gradient[:-1] * xstep/2
            densities[1:-1] = (midpoint_densities + np.roll(midpoint_densities, -1))[:-1]/2
            
            #Recalculating pressures
            pressures[1:-1] = pressure(densities[1:-1])
            
            # Pulse Tube Inlet boundary condition
            pressures[0] = pulse_tube_inlet_pressure(t + tstep)
            densities[0] = pulse_tube_inlet_density(t + tstep)
            
            # # Calculating for flow into buffer tank
            # buffer_tank_density += mass_flows[-1] / self.buffer_tank_volume
            # buffer_tank_pressure = buffer_tank_density * self.specific_gas_constant * mean_temperature
            
            # Buffer Tank Inlet Boundary condition
            pressures[-1] = mean_pressure
            densities[-1] = mean_density
            
            # Logging inlet mass flow
            pulse_tube_inlet_mass_flowrate[i] = velocities[0]*densities[0]*self.flow_area
            
            # Advancing iteration counter
            i += 1
        
        plt.figure("P")
        plt.plot(x_positions, pressures)
        plt.figure("U")
        plt.plot(x_positions, velocities)
        plt.figure("mdot")
        plt.plot(phase_angles_deg, pulse_tube_inlet_mass_flowrate[-t_divisions:])
        plt.show()
        

if __name__ == "__main__":
    inertance_tube = LaminarInertanceTube(length=1.689, diameter=1.016E-3, buffer_tank_volume=0.01)
    print(2.5E+6 * (1.1 - 1)/(1.1 + 1) * 1E-5)
    inertance_tube.simulate(frequency=55, mean_pressure=2.5E+6, pressure_ratio=1.4, mean_temperature=300, polytropic_index=1.67, periods=10)