import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

class TurbulentInertanceTube:
    def __init__(self, length:float, diameter:float, buffer_tank_volume:float = 1, absolute_roughness:float = 0.15E-3):
        self.length = length
        self.diameter = diameter
        self.buffer_tank_volume = buffer_tank_volume
        self.flow_area = np.pi/4 * diameter**2
        
        self.absolute_viscosity = 2.0012e-05
        self.specific_gas_constant = 2077.264394
        #self.friction_factors = (-2*np.log10(absolute_roughness/(3.7*self.diameter)))**-2
    
    def simulate(self, frequency:float, mean_pressure:float, pressure_ratio:float, mean_temperature:float, polytropic_index:float, x_divisions:int=101, t_divisions:int=5000, periods:int=20, plot:bool = True):
        # Calculating some constants
        pressure_amplitude = mean_pressure * (pressure_ratio - 1)/(pressure_ratio + 1)
        mean_density = mean_pressure/(mean_temperature * self.specific_gas_constant)
        polytropic_coefficient = (mean_pressure/mean_density**polytropic_index)
        period = 1 / frequency
        #friction_coefficient = self.friction_factors/(2 * self.diameter) #0.1582*self.absolute_viscosity**(1/4)*self.diameter**-5
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
        mass_fluxes = np.zeros(x_divisions, dtype=float)
        pressures = np.full(x_divisions, mean_pressure, dtype=float)
        densities = np.full(x_divisions, mean_density, dtype=float)
        reynolds_numbers = np.zeros(x_divisions, dtype=float)
        midpoint_densities = np.full(x_divisions - 1, mean_density, dtype=float)
        x_positions = np.linspace(0, self.length, x_divisions, dtype=float)
        times = np.linspace(0, period * periods, t_divisions*periods, dtype=float, endpoint=True)
        phase_angles_deg = np.linspace(0, 360, t_divisions, dtype=float, endpoint=True)
        buffer_tank_pressure = mean_pressure
        buffer_tank_density = mean_density
        friction_factors = np.zeros(x_divisions, dtype=float)
        
        pulse_tube_inlet_mass_flowrate = np.zeros(periods * t_divisions, dtype=float)
        pulse_tube_inlet_velocity = np.zeros(periods * t_divisions, dtype=float)
        
        transition_reynolds_number = 4000
            
        # Iteration
        i = 0
        for t in times:
            # if i == 3:
            #     break
            
            # first x derivative arrays
            velocity_x_derivative = np.gradient(velocities, xstep, edge_order=2)
            density_x_derivative = np.gradient(densities, xstep, edge_order=2)
            pressure_x_derivative = np.gradient(pressures, xstep, edge_order=2)
            
            # Blasius Friction factor
            # friction_coefficient = np.where(reynolds_numbers < 4000,
            #                                 32/(reynolds_numbers * self.diameter),
            #                                 0.1582*(self.absolute_viscosity/self.diameter**5)**(1/4)*(densities*np.abs(velocities))**(-1/4))
            #Friction factor calculation
            friction_factors = np.where((reynolds_numbers > 0) & (reynolds_numbers < transition_reynolds_number), 64/reynolds_numbers, np.where(reynolds_numbers==0, 0, 0.3164*reynolds_numbers**(-1/4)))
            #friction_factors = np.where(reynolds_numbers == 0, 0, friction_factors)
            
            # first t derivative arrays
            velocity_t_derivative = -1/densities*pressure_x_derivative - np.sign(velocities) * friction_factors/(2*self.diameter) * velocities**2
            density_t_derivative = -(densities*velocity_x_derivative + velocities*density_x_derivative)
            # Boundary conditions
            density_t_derivative[0] = inlet_density_t_derivative(t)
            density_t_derivative[-1] = 0
            pressure_t_derivative = polytropic_coefficient*polytropic_index*densities**(polytropic_index - 1)*density_t_derivative
            # Boundary conditions
            pressure_t_derivative[0] = inlet_pressure_t_derivative(t)
            pressure_t_derivative[-1] = 0
            #misc t derivatives
            mass_flux_t_derivative = density_t_derivative*velocities + velocity_t_derivative*densities
            reynolds_number_t_derivative = self.diameter/self.absolute_viscosity * mass_flux_t_derivative
            friction_factors_t_derivative = np.where((reynolds_numbers > 0) & (reynolds_numbers < transition_reynolds_number), -64 * reynolds_number_t_derivative * reynolds_numbers**-2, np.where( reynolds_numbers == 0, 0, -0.3164/4 * reynolds_number_t_derivative * reynolds_numbers**(-5/4)))
            # friction_coefficient_t_derivative = np.where(reynolds_numbers < 4000,
            #                                              0,
            #                                              -0.03955*(self.absolute_viscosity/self.diameter**5)**(1/4) * (density_t_derivative*velocities + velocity_t_derivative*densities) * (densities*np.abs(velocities))**-5/4)
            
            # xt derivative arrays
            pressure_xt_derivative = np.gradient(pressure_t_derivative, xstep, edge_order=2)
            
            # # second t derivative arrays
            """ Quadratic
            velocity_tt_derivative = (densities**-2 * density_t_derivative*pressure_x_derivative - densities**-1 * pressure_xt_derivative
                                      - np.sign(velocities) * friction_coefficient * (densities**-2*density_t_derivative*velocities**2 - 2*densities**-1 * velocity_t_derivative * velocities))
            """
            # Assuming blasius correlation
            # velocity_tt_derivative =  (densities**-2 * density_t_derivative*pressure_x_derivative - densities**-1 * pressure_xt_derivative
            #                             - np.sign(velocities) * ((friction_coefficient_t_derivative * densities**-1 - friction_coefficient * density_t_derivative * densities**-2) * velocities**2 + 2* velocity_t_derivative * friction_coefficient/densities * velocities))
            velocity_tt_derivative = (
                                        densities**-2 * density_t_derivative * pressure_x_derivative - densities**-1 * pressure_xt_derivative
                                      - np.sign(velocities)*(friction_factors_t_derivative*velocities**2 + 2* velocities*velocity_t_derivative*friction_factors)
                                      )
            
            # Calculating new densities, excluding boundaries
            #velocity_changes = velocity_t_derivative * tstep + 0.5 * velocity_tt_derivative * tstep **2
            # velocity_change_limit = 1
            # velocities = velocities + np.where(np.abs(velocity_changes) > velocity_change_limit, velocity_change_limit, velocity_changes)
            velocities += velocity_t_derivative * tstep + 0.5 * velocity_tt_derivative * tstep **2
            mass_fluxes = velocities * densities
            mass_flows = (mass_fluxes * self.flow_area) * tstep
            midpoint_densities += (mass_flows - np.roll(mass_flows, -1))[:-1]/vstep

            # Calculaing new densities from midpoints
            #midpoint_density_gradient = np.gradient(midpoint_densities, xstep, edge_order=2)
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
            
            # Reynolds number updating
            reynolds_numbers = densities * np.abs(velocities) * self.diameter / self.absolute_viscosity
            
            # Logging inlet mass flow
            pulse_tube_inlet_mass_flowrate[i] = velocities[0]*densities[0]*self.flow_area
            pulse_tube_inlet_velocity[i] = velocities[0]
            
            # Advancing iteration counter
            i += 1
        
        # maximum_flow_arg = np.argmax(pulse_tube_inlet_mass_flowrate[-t_divisions:])
        # minimum_flow_arg = np.argmin(pulse_tube_inlet_mass_flowrate[-t_divisions:])
        # maximum_flow_phase_angle = phase_angles_deg[maximum_flow_arg]
        # minimum_flow_phase_angle = phase_angles_deg[minimum_flow_arg]
        # root_1_angle = phase_angles_deg[-t_divisions:-t_divisions//2][np.argmin(np.abs(pulse_tube_inlet_velocity[-t_divisions:-t_divisions//2]))]
        # root_2_angle = phase_angles_deg[-t_divisions//2:][np.argmin(np.abs(pulse_tube_inlet_velocity[-t_divisions//2:]))]
        
        #phase_lag = float((root_1_angle + root_2_angle - 180 + maximum_flow_phase_angle - 90 + minimum_flow_phase_angle - 270)/4)
        #phase_lag = (maximum_flow_phase_angle - 90 + minimum_flow_phase_angle - 270)/2
        
        def sinusoidal_approximation(theta, amplitude, phase_shift):
            return amplitude*np.sin(np.deg2rad(theta - phase_shift))
        velocity_magnitude, phase_lag = curve_fit(f=sinusoidal_approximation, xdata=phase_angles_deg, ydata=pulse_tube_inlet_velocity[-t_divisions:], p0=[np.max(velocities), 0])[0]
        velocity_residual = float(np.abs(pulse_tube_inlet_velocity[-1] - pulse_tube_inlet_velocity[-t_divisions-1]))
        # phase_lag = (maximum_flow_phase_angle - 90 + minimum_flow_phase_angle - 270)/2
        #print(phase_lag)
        #return phase_lag
        if plot:
            fig, ax1 = plt.subplots()
            ax2 = ax1.twinx()
            ax1.plot(phase_angles_deg, pulse_tube_inlet_velocity[-t_divisions:], label="Pulse Tube Inlet Velocity", color='g', linestyle="-")
            ax1.plot(phase_angles_deg, sinusoidal_approximation(phase_angles_deg, velocity_magnitude, phase_lag), label="Pulse Tube Inlet Velcoity (Sinusoidal Fit)", color='g', linestyle="--")
            ax2.plot(phase_angles_deg, 1E-6 * pulse_tube_inlet_pressure(times[-t_divisions:]), label="Pulse Tube Inlet Pressure", color = 'r', linestyle="-")
            
            ax1.set_xlabel("Phase Angle (degrees)")
            ax1.set_ylabel('Flow Velocity (m/s)', color='g')
            ax2.set_ylabel('Pressure (MPa)', color='r')
            fig.legend()
            plt.tight_layout()
        return float(velocity_magnitude), float(phase_lag), velocity_residual
            
            #plt.show()
            # plt.figure("P")
            # plt.plot(x_positions, pressures)
            # plt.figure("U")
            # plt.plot(x_positions, velocities)
            # plt.figure("mdot")
            # plt.plot(phase_angles_deg, pulse_tube_inlet_mass_flowrate[-t_divisions:])
            # plt.show()
        
        #return (phase_lag, velocity_residual)
        

if __name__ == "__main__":
    # inertance_tube = TurbulentInertanceTube(length=1.689, diameter=1.016E-3)
    inertance_tube = TurbulentInertanceTube(length=1.219, diameter=1.574E-3)
    #print(inertance_tube.simulate(frequency=55, mean_pressure=2.5E+6, pressure_ratio=1.4, mean_temperature=300, polytropic_index=0.8))
    #plt.show()
    #pass
    pressure_ratios = [1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4]
    frequencies = [30, 40, 50, 60, 70, 80, 90]
    pr_results = []
    f_results = []
    
    for pressure_ratio in pressure_ratios:
        print(pressure_ratio)
        pr_results.append(inertance_tube.simulate(frequency=55, mean_pressure=2.5E+6, pressure_ratio=pressure_ratio, mean_temperature=300, polytropic_index=0.9, plot=False))
    # #print(pr_results)
    # print("")
    
    for result in pr_results:
        print(*result)
    print("")
    
    # for frequency in frequencies:
    #     print(frequency)
    #     f_results.append(inertance_tube.simulate(frequency=frequency, mean_pressure=2.5E+6, pressure_ratio=1.25, mean_temperature=300, polytropic_index=1.663732394, plot=False))
    # print("")
    # for result in f_results:
    #     print(*result)
    
    # print(pr_results)
    # print(f_results)