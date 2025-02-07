import numpy as np
import random
import matplotlib.pyplot as plt
import geopandas as gpd
import pandas as pd
import plotly.graph_objects as go

#############################################################
"""Assigns all the actors and their specific attributes"""
#############################################################


class Actor:
    def __init__(self, type):
        self.type = type

class Hexagon:
    """Hexagons have meta attributes"""
    def __init__(self, x, y):
        self.coordinates = (x, y) # coordinates for CA
        self.real_coordinates = None # coordinates for map graph
        self.actor = None
        self.pm10 = 0
        self.pm25 = 0
        self.temperature = random.uniform(15, 35) # not used yet

class Building(Actor):
    def __init__(self):
        super().__init__('building')

class Water(Actor):
    def __init__(self):
        super().__init__('water')

class FreeSpace(Actor):
    def __init__(self):
        super().__init__('free')

class OutBound(Actor):
    """possible due to square grid and not square zipcodes"""
    def __init__(self):
        super().__init__('OoB')

class Road(Actor):
    """
    Now amount of cars are based on maximum capacity according to: 
    https://www.researchgate.net/publication/263504490_A_cellular_automaton_model_for_freeway_traffic 

    At a 5-meter distance from a vehicle:
	PM10 concentration: 100 µg/m³
	PM2.5 concentration: 20 µg/m³
    https://pubmed.ncbi.nlm.nih.gov/11347914/ """
    def __init__(self, direction):
        super().__init__('road')
        self.direction = direction
        self.car = random.random() < 0.05
        self.exhaust_PM10 = 100
        self.exhaust_PM25 = 20

class Tree(Actor):
    """
    Average urban trees: The absorption efficiency can range from 5% to 20%, depending on species and conditions.
    Absorbtion can also be based on the leaf surface area which can be calculated from the height and species:
    https://www.sciencedirect.com/science/article/pii/S0378112717301238 
    """
    def __init__(self, initial_height):
        super().__init__('tree')
        self.height_tree = initial_height
        self.absorption_efficiency = 0.1
        self.leaf_surface_area = 2 * (initial_height ** 2.2) # leaf surface area of Iep 10m 

#############################################################
#############################################################
"""Cellular automata, interactions and simulation methods"""
#############################################################

class UrbanEnvironmentSimulation:
    def __init__(self, width=20, height=20, data = False, random = True):
        self.width = width
        self.height = height
        self.data = data
        self.congestion_prob = 0.05
        self.random = random
        """Random grid and grid from data can both be initialized this way"""
        if random:
            self.grid = self.initialize_grid()
        else:
            self.grid = self.grid_from_dataset(data, height, width)

        self.colors = {
            'free': (0.88, 0.88, 0.88, 1),     # Light gray
            'road': (0.5, 0.5, 0.5, 1),         # Dark gray
            'water': (0.31, 0.69, 1.0, 1),      # Light blue
            'building': (0.63, 0.36, 0.24, 1),   # Brown
            'tree': (0.13, 0.55, 0.13, 1),       # Forest green
            'OoB': (0, 0, 0, 0)                 # transparent
        }

    def initialize_grid(self, n_trees = 6):
        """Produces initial random grid with all actors"""
        grid = [[Hexagon(x, y) for y in range(self.height)] for x in range(self.width)]
        n_tree_pos = [i * (self.width // n_trees)+1 for i in range(n_trees)]

        middle_row = self.height // 2
        
        for x in range(self.width):
            for y in range(self.height):
                if y < 4:
                    grid[x][y].actor = Water()
                
                elif y == middle_row or y == middle_row - 1:
                    if x == 0 or x == self.width - 1:
                        grid[x][y].actor = Road(direction='intersection')
                    else:
                        grid[x][y].actor = Road(direction='straight')
                
                elif middle_row < y < self.height - 2:
                    grid[x][y].actor = Building()
                    grid[x][y].pm10 = 0
                    grid[x][y].pm25 = 0
                
                elif y == 6:
                    if x in n_tree_pos:
                        grid[x][y].actor = Tree(initial_height=10)  
                    else:
                        grid[x][y].actor = FreeSpace() 
                
                else:
                    grid[x][y].actor = FreeSpace()  

        return grid


    def grid_from_dataset(self, df, h, w):
        """Produces initial grid from preproccesed real data"""
        self.height = h
        self.width = w

        grid = [[Hexagon(x, y) for y in range(self.height)] for x in range(self.width)]

        for x in range(self.width):
            for y in range(self.height):
                current_hex = df[(df.x == x) & (df.y == y)]
                try:
                    grid[x][y].real_coordinates = current_hex.geometry.iloc[0]
                    if current_hex.actor.iloc[0] == 'tree':
                        grid[x][y].actor = Tree(initial_height=10)
                    elif current_hex.actor.iloc[0] == 'road' or current_hex.actor.iloc[0] == 'intersection':
                        grid[x][y].actor = Road(direction='straight')
                    elif current_hex.actor.iloc[0] == 'building':
                        grid[x][y].actor = Building()
                    elif current_hex.actor.iloc[0] == 'water':
                        grid[x][y].actor = Water()
                    elif current_hex.actor.iloc[0] == None:
                        grid[x][y].actor = FreeSpace()
                except:
                    grid[x][y].actor = OutBound()

                if grid[x][y].actor.type != 'building' and grid[x][y].actor.type != 'OoB':
                    grid[x][y].pm10 = random.uniform(22, 59) #Starts with random pm10 value distribution based on (https://aqicn.org/city/amsterdam/)
                    grid[x][y].pm25 = random.uniform(69, 154) #Starts with random pm2.5 value distribution based on (https://aqicn.org/city/amsterdam/)
        return grid

    def PM_diffusion(self, x, y):
        """diffusion based on Ficks Laws
        https://www.geeksforgeeks.org/ficks-law-of-diffusion/#what-is-ficks-law-of-diffusion
        
        First law: is a scientific principle that describes the movement of particles from an area of high concentration to an area of low concentration.
        Second law: describes the time-dependent behavior of the concentration profile during diffusion
        """

        directions = [(-1, -1), (-1, 0), (-1, 1),
                          (0, -1), (0, 1),
                          (1, -1), (1, 0), (1, 1)]
        current_cell = self.grid[x][y]
        
        neighbour_cords_pm10 = []
        neighbour_dif_pm10 = []

        neighbour_cords_pm25 = []
        neighbour_dif_pm25 = []

        """Saves cell attributes (location and concentration difference) if concentration is higher than neighbour"""
        for dx, dy in directions:
            neighbor_x, neighbor_y = x + dx, y + dy
            if 0 <= neighbor_x < self.width and 0 <= neighbor_y < self.height:
                neighbor_cell = self.grid[neighbor_x][neighbor_y]
                if neighbor_cell.actor.type != 'building' and neighbor_cell.actor.type != 'OoB':
                    if current_cell.pm10 > neighbor_cell.pm10:
                        neighbour_cords_pm10.append((neighbor_x, neighbor_y))
                        dif = current_cell.pm10 - neighbor_cell.pm10
                        neighbour_dif_pm10.append(dif)

                    if current_cell.pm25 > neighbor_cell.pm25:
                        neighbour_cords_pm25.append((neighbor_x, neighbor_y))
                        dif = current_cell.pm25 - neighbor_cell.pm25
                        neighbour_dif_pm25.append(dif)
        
        """Lower concentration neighbours get more PM from the current cell (percentage wise)"""
        if len(neighbour_dif_pm10) > 0:
            neighbour_dif_pm10 = np.array(neighbour_dif_pm10) / np.sum(neighbour_dif_pm10)
        if len(neighbour_dif_pm25) > 0:
            neighbour_dif_pm25 = np.array(neighbour_dif_pm25) / np.sum(neighbour_dif_pm25)


        for (neighbor_x, neighbor_y), dif_prob in zip(neighbour_cords_pm10, neighbour_dif_pm10):
            neighbor_cell = self.grid[neighbor_x][neighbor_y]
            amount_pm10 = (current_cell.pm10 - neighbor_cell.pm10) /2 * dif_prob
            current_cell.pm10 -= amount_pm10
            neighbor_cell.pm10 += amount_pm10

        for (neighbor_x, neighbor_y), dif_prob in zip(neighbour_cords_pm25, neighbour_dif_pm25):
            neighbor_cell = self.grid[neighbor_x][neighbor_y]
            amount_pm25 = (current_cell.pm25 - neighbor_cell.pm25) /2 * dif_prob
            current_cell.pm25 -= amount_pm25
            neighbor_cell.pm25 += amount_pm25

        
    def update(self):
        for x in range(self.width):
            for y in range(self.height):
                hex_cell = self.grid[x][y]

                if hex_cell.actor.type == 'tree':
                    '''tree absorption of PM10 and PM2.5'''
                    pm10_absorption =  hex_cell.pm10 * hex_cell.actor.absorption_efficiency * (1+random.random())
                    hex_cell.pm10 -= pm10_absorption
                    
                    pm25_absorption =  hex_cell.pm25 * hex_cell.actor.absorption_efficiency * (1+random.random())
                    hex_cell.pm25 -= pm25_absorption
                elif hex_cell.actor.type == 'road':
                    '''car pollution'''
                    if hex_cell.actor.car:
                        hex_cell.pm10 += hex_cell.actor.exhaust_PM10
                        hex_cell.pm25 += hex_cell.actor.exhaust_PM25
                    hex_cell.actor.car = random.random() < self.congestion_prob

                if hex_cell.actor.type != 'building' and hex_cell.actor.type != 'OoB':
                    """Diffusion set in motion and Random PM is added by other unspecified factors (planes, factories etc.)"""
                    self.PM_diffusion(x, y)
                    hex_cell.pm10 = hex_cell.pm10/2 if hex_cell.pm10 > 150 else hex_cell.pm10
                    hex_cell.pm25 = hex_cell.pm25/2 if hex_cell.pm25 > 170 else hex_cell.pm25
                    
        return self.grid

    def return_pm_avg(self, pm10 = False):
        if pm10:
            return np.mean([value.pm10 for array in self.grid for value in array if value.pm10 > 0])
        else:
            return np.mean([value.pm25 for array in self.grid for value in array if value.pm25 > 0])
    
    def simulate(self, days=1, n_sim= 3):
        self.days = days
        self.n_sim = n_sim
        n_day = 288
        period_amount = days * n_day
        self.congestion_probs = pd.read_csv('/Users/Mick/Desktop/1.3 Data Systems Project/data/traffic_congestion_probabilities.csv')
        
        pm10_list = []
        pm25_list = []

        for _ in range(n_sim):
            if not self.random:
                self.grid = self.grid_from_dataset(self.data, self.height, self.width)
            else:
                self.grid = self.initialize_grid()

            pm10_current_sim = [self.return_pm_avg(pm10=True)]
            pm25_current_sim = [self.return_pm_avg(pm10=False)]

            for n_time in range(period_amount*2):
                self.congestion_prob = self.congestion_probs.at[n_time % n_day, 'Congestion_Probability']
                self.update()

            for n_time in range(period_amount):
                self.congestion_prob = self.congestion_probs.at[n_time % n_day, 'Congestion_Probability']
                self.update()
                pm10_current_sim.append(self.return_pm_avg(pm10=True))
                pm25_current_sim.append(self.return_pm_avg(pm10=False))
                # if n_time % 50 == 0:
                # self.show_heatmap()
            pm10_list.append(pm10_current_sim)
            pm25_list.append(pm25_current_sim)
        
        self.plot_simulation_results(pm10_list, pm25_list)


    #############################################################
    #############################################################
    """Plotting methods for heatmaps, random and data based simulations"""
    #############################################################

    def plot_simulation_results(self, pm10_list, pm25_list):
        pm10_array = np.array(pm10_list)
        pm25_array = np.array(pm25_list)


        pm10_mean = pm10_array.mean(axis=0)
        pm10_std = pm10_array.std(axis=0) * 10
        pm25_mean = pm25_array.mean(axis=0)
        pm25_std = pm25_array.std(axis=0)  * 10

        self.congestion_probs['Timestamp'] = pd.to_datetime(self.congestion_probs['Timestamp'])
        self.congestion_probs['Timestamp'] = self.congestion_probs['Timestamp'].apply( lambda x: x.replace(year=2024, month=3, day=14))
        time_steps = self.congestion_probs['Timestamp'][:-1]


        fig = go.Figure()
        
        # Plot PM10
        fig.add_trace(go.Scatter(
            x=time_steps, y=pm10_mean, mode='lines', name='PM10 Mean',
            line=dict(color='blue'),
            error_y=dict(type='data', array=pm10_std, visible=True, color='blue', thickness=0.5)
        ))

        # Plot PM25
        fig.add_trace(go.Scatter(
            x=time_steps, y=pm25_mean, mode='lines', name='PM25 Mean',
            line=dict(color='red'),
            error_y=dict(type='data', array=pm25_std, visible=True, color='red', thickness=0.5)
        ))

        fig.update_layout(
            title=f'PM10 and PM25 Simulation Results of {self.days} Day(s) and {self.n_sim} Simulations',
            xaxis_title='Hour',
            yaxis_title='PM Concentration in µg/m³',
            showlegend=True
        )

        fig.show()


    def plot_grid(self, figsize=(10, 8)):
        plt.figure(figsize=figsize)
        color_matrix = np.zeros((self.height, self.width, 4))
        
        for x in range(self.width):
            for y in range(self.height):
                cell_type = self.grid[x][y].actor.type
                color_matrix[y, x] = self.colors[cell_type]
        
        plt.imshow(color_matrix)
        
        plt.grid(False)
        

        legend_elements = [plt.Rectangle((0, 0), 1, 1, facecolor=colour, label=type_name)
                        for type_name, colour in self.colors.items()]
        
        plt.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')
        

        plt.title('Zipcode 1074 in Amsterdam')
        plt.xlabel('')
        plt.ylabel('')
        
        plt.tight_layout()
        
        # plt.draw()  
        # plt.pause(2)  
        # plt.close()
        plt.show()

    def show_heatmap(self, figsize=(10, 8)):
        plt.figure(figsize=figsize)
        pm10_values = np.array([[self.grid[x][y].pm10 for x in range(self.width)] for y in range(self.height)])
        plt.imshow(pm10_values, cmap='YlGnBu', interpolation='nearest')
        plt.colorbar(label='PM10 Concentration (µg/m³)')
        plt.title('Heatmap of PM10 Concentration')
        plt.draw()  
        plt.pause(0.2)  
        plt.close()

#############################################################
#############################################################

if __name__ == "__main__":

    test_place = gpd.read_file('/Users/Mick/Desktop/1.3 Data Systems Project/data/amstel_dijk_polygon.geojson')


    # sim = UrbanEnvironmentSimulation(max(test_place.x)+1, max(test_place.y)+1, data = test_place, random=False)
    sim = UrbanEnvironmentSimulation(20,20, random=True)
    
    sim.plot_grid()

    # sim.simulate(days=1, n_sim= 10)

    
