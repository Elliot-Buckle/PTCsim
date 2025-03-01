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
    
    def simulate(self, frequency:float, mean_pressure:float, pressure_ratio:float, mean_temperature:float, polytropic_index:float, x_divisions:int=100, t_divisions:int=100):
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
            # Pulse Tube Inlet boundary condition
            pressures[0] = pulse_tube_inlet_pressure(t)
            densities[0] = pulse_tube_inlet_density(t)
            
            # Buffer Tank Inlet Boundary condition
            pressures[-1] = mean_pressure
            densities[-1] = mean_density
            
            # X Difference arrays
            velocity_xgradient = first_derivative(velocities, xstep)
            density_xgradient = first_derivative(densities, xstep)
            pressure_xgradient = first_derivative(pressures, xstep)
            # mass_flux_xgradient = first_derivative(mass_fluxes, xstep)
            
            # Applying pulse tube inlet flow acceleration
            if velocities[0] == 0:
                velocities[0] += tstep * -(pressure_xgradient[0])/densities[0]
            else:
                velocities[0] += tstep * (-1/densities[0] * (
                    polytropic_coefficient * polytropic_index * densities[0]**(polytropic_index - 1)
                    * (-1/velocities[0]*(inlet_density_t_derivative(t) + densities[0] * velocity_xgradient[0]))
                    + friction_coefficient * velocities[0]))
                
            # Applying buffer tank inlet flow acceleration
            if velocities[-1] == 0:
                velocities[-1] += tstep * -1/mean_density * pressure_xgradient[-1]
            else:
                velocities[-1] += tstep * (polytropic_coefficient * polytropic_index * mean_density**(polytropic_index - 1)*velocity_xgradient[-1]/velocities[-1] - friction_coefficient*velocities[-1]/mean_density)
                
            # Now the boundary conditions have been applied, the flow in the rest of the inertance tube is evolved
            velocity_change = tstep * -1/densities*(pressure_xgradient + friction_coefficient*velocity_xgradient)
            density_change = tstep * -(densities*velocity_xgradient + velocities*density_xgradient)
            
            # The changes at the inlet and outlet are set to zero, since they were calculated earlier
            velocity_change[0] = 0
            velocity_change[-1] = 0
            density_change[0] = 0
            density_change[-1] = 0
            
            # The calculated density and velocity changes are added to their respective arrays
            velocities += velocity_change
            densities += density_change
            
            # Pressures are recaluclated from the new densities using a polytropic equation of state
            pressures = pressure(densities)
            
            i+=1
            if i == 2:
                break
        
        plt.plot(x_positions, pressures)
        plt.show()
        

if __name__ == "__main__":
    inertance_tube = LaminarInertanceTube(length=1.689, diameter=1.016E-3)
    inertance_tube.simulate(55, 2.5E+6, 1.1, 300, 1)