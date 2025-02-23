import numpy as np
import matplotlib.pyplot as plt

class InertanceTube:
    def __init__(self, length:float, diameter:float, mean_pressure:float, pressure_amplitude:float, frequency:float, mean_temperature:float, absolute_roughness:float = 0):
        self.length = length
        self.diameter = diameter
        self.mean_pressure = mean_pressure
        self.pressure_amplitude = pressure_amplitude
        self.frequency = frequency
        self.period = 1/frequency
        self.angular_velocity = self.frequency * 2 * np.pi
        self.mean_temperature = mean_temperature
        self.absolute_roughness = absolute_roughness
        
        self.absolute_viscosity = 1.96E-5
        self.specific_gas_constant = 2077.1
        
        self.mean_density = self.mean_pressure / (self.specific_gas_constant * self.mean_temperature)
        
    def simulate(self, polytropic_index:float = 1.67, x_divisions:int = 10, periods:int = 10, period_divisions:int=1000):
        polytropic_constant = self.mean_pressure/self.mean_density**polytropic_index
        self.pressures = np.full(x_divisions, self.mean_pressure, float)
        self.densities = np.full(x_divisions, self.mean_density, float)
        self.velocities = np.zeros(x_divisions, float)
        self.reynolds = np.zeros(x_divisions, float)
        self.pressure_gradient = np.zeros(x_divisions, float)
        self.friction_factor = 0.025
        self.xpoints = np.linspace(0, self.length, x_divisions)
        
        self.xstep = self.length / x_divisions
        #tstep = self.xstep / np.sqrt(1.67*self.specific_gas_constant*self.mean_temperature)
        # print(tstep)
        # tstep = 1/(self.frequency*10000)
        #tstep = self.period/period_divisions
        #self.times = np.arange(0, self.period*periods, tstep)
        self.times = np.linspace(0, self.period, period_divisions)
        tstep = self.times[1]
        
        self.inlet_mass_flows = np.zeros(period_divisions)
        self.inlet_pressures = np.zeros(period_divisions)
        for period in range(periods):
            for step in range(period_divisions):
                self.pressures[0] = self.mean_pressure + self.pressure_amplitude*np.sin(self.angular_velocity * self.times[step])
                self.pressures[-1] = self.mean_pressure
                
                self.pressure_gradient = self.first_derivative(self.pressures, self.xstep)
                
                # density_change = np.zeros(x_divisions)
                # velocity_change = np.zeros(x_divisions)
                average_pressure_gradient = self.pressure_gradient
                average_velocity = self.velocities
                average_density = self.densities 
                
                for i in range(10):
                    velocity_change = (-1/(average_density) * average_pressure_gradient - np.sign((average_velocity)) * average_velocity**2 / (2 * self.diameter)
                                    * self.friction_factor
                                    ) * tstep
                    
                    #print(velocity_change[0:4])
                    
                    density_change = - self.first_derivative(average_density*average_velocity, self.xstep) * tstep
                    average_pressure = polytropic_constant*(average_density)**polytropic_index
                    average_pressure[0] = self.pressures[0]
                    average_pressure[-1] = self.mean_pressure
                    
                    average_pressure_gradient = (self.first_derivative(average_pressure, self.xstep))
                    average_velocity = self.velocities + velocity_change/2
                    average_density = self.densities + density_change/2
                
                self.pressures = average_pressure
                self.velocities += velocity_change
                self.densities += density_change
                self.reynolds = self.densities*self.velocities*self.diameter/self.absolute_viscosity
                self.inlet_mass_flows[step] = np.pi/4 * self.diameter**2 * self.densities[0] * self.velocities[0]
                self.inlet_pressures[step] = self.pressures[0]
                #break
            print(abs(self.inlet_mass_flows[0] - self.inlet_mass_flows[-1]))
            
            #self.reynolds = self.densities*self.velocities*self.diameter/self.absolute_viscosity
        # plt.figure("inlet pressure vs time")
        # plt.plot(self.times, self.inlet_pressures)
        print(np.max(self.reynolds))
        plt.figure("inlet mass flow vs time")
        plt.plot(360*self.times/self.period, self.inlet_mass_flows)
        plt.figure("pressure vs x")
        plt.plot(self.xpoints, self.pressures*10**-5)
        plt.figure("velocity vs x")
        plt.plot(self.xpoints, self.velocities)
        #plt.plot(self.velocities)
        plt.show()
    
    def first_derivative(self, array:np.ndarray, step:float):
        gradient = (np.roll(array, -1) - np.roll(array, 1))/(2*step)
        gradient[0] = (-3/2 * array[0] + 2 * array[1] - 1/2 * array[2])/step
        gradient[-1] = (3/2 * array[-1] - 2 * array[-2] + 1/2*array[-3])/step
        return gradient
            

if __name__ == "__main__":
    inertance_tube = InertanceTube(length=1.689, diameter=1.016E-3, mean_pressure=2.5E+6, pressure_amplitude=0.2777777778E+6, frequency=55, mean_temperature=300)
    inertance_tube.simulate()