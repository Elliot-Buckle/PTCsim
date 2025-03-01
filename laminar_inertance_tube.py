import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root
from miscellaneous import first_derivative

class LaminarInertanceTube:
    def __init__(self, length:float, diameter:float, absolute_roughness:float = 0):
        self.length = length
        self.diameter = diameter
        
        self.absolute_viscosity = 1.96E-5
        self.specific_gas_constant = 2077.1
    
    def simulate(self, frequency:float, mean_pressure:float, pressure_ratio:float, mean_temperature:float, polytropic_index:float, x_divisions:int=10, t_divisions:int=1000):
        # Calculating some constants
        pressure_amplitude = mean_pressure * (pressure_ratio - 1)/(pressure_ratio + 1)
        mean_density = mean_pressure/(mean_temperature * self.specific_gas_constant)
        polytropic_coefficient = (mean_pressure/mean_density**polytropic_index)
        period = 1 / frequency
        friction_coefficient = 32*self.absolute_viscosity/self.diameter**2
        tstep = period / t_divisions
        xstep = self.length / x_divisions
        
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
        mass_fluxes = np.zeros(x_divisions, dtype=float)
        x_positions = np.linspace(0, self.length, x_divisions, dtype=float)
        times = np.linspace(tstep, period + tstep, t_divisions, dtype=float)
        phase_angles_deg = np.linspace(0, 360, t_divisions, dtype=float)
        
        # Iteration
        i = 0
        for t in times:
            # if i == 1:
            #     break
            
            # Pulse Tube Inlet boundary condition
            pressures[0] = pulse_tube_inlet_pressure(t)
            densities[0] = pulse_tube_inlet_density(t)
            
            # Buffer Tank Inlet Boundary condition
            pressures[-1] = mean_pressure
            densities[-1] = mean_density
            
            # first x derivative arrays
            velocity_x_derivative = first_derivative(velocities, xstep)
            density_x_derivative = first_derivative(densities, xstep)
            pressure_x_derivative = first_derivative(pressures, xstep)
            
            # first t derivative arrays
            velocity_t_derivative = -1/densities*(pressure_x_derivative + friction_coefficient * density_x_derivative)
            density_t_derivative = -(densities*velocity_x_derivative + velocities*density_x_derivative)
            pressure_t_derivative = polytropic_coefficient*polytropic_index*densities**(polytropic_index - 1)*density_t_derivative
            
            # xt derivative arrays
            velocity_xt_derivative = first_derivative(velocity_t_derivative, xstep)
            density_xt_derivative = first_derivative(density_t_derivative, xstep)
            pressure_xt_derivative = first_derivative(pressure_t_derivative, xstep)
            
            # second t derivative arrays
            velocity_tt_derivative = (
                                        densities**-2 * density_t_derivative*pressure_x_derivative
                                      - densities**-1 * pressure_xt_derivative
                                      - friction_coefficient*velocity_t_derivative
                                      )
            density_tt_derivative = (
                                     - density_t_derivative*velocity_x_derivative
                                     - densities*velocity_xt_derivative
                                     - velocity_t_derivative*density_x_derivative
                                     - velocities*density_xt_derivative
                                     )
            
            # Calculating new densities, excluding boundaries
            densities[1:-1] += density_t_derivative[1:-1] * tstep + 0.5 * density_tt_derivative[1:-1] * tstep**2
            velocities += velocity_t_derivative * tstep + 0.5 * velocity_tt_derivative * tstep **2
            
            #Recalculating pressures
            pressures = pressure(densities)
            
            #print(velocity_t_derivative[0], velocity_tt_derivative[0])
            
            i += 1
        
        plt.figure("P")
        plt.plot(x_positions, pressures)
        plt.figure("U")
        plt.plot(x_positions, velocities)
        plt.show()
        

if __name__ == "__main__":
    inertance_tube = LaminarInertanceTube(length=1.689, diameter=1.016E-3)
    print(2.5E+6 * (1.1 - 1)/(1.1 + 1) * 1E-5)
    inertance_tube.simulate(55, 2.5E+6, 1.1, 300, 1)