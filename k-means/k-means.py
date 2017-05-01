import numpy as np
import pandas as pd

# Use the bokeh library for visualization
from bokeh.plotting import figure, ColumnDataSource, output_file, show
from bokeh.models import CategoricalColorMapper
from bokeh.layouts import Column, Row
from bokeh.models.widgets import Tabs, Panel
from bokeh.palettes import Category10

class kmeans():
    def __init__(self, df, max_iter, epochs, num_centers):
        self.df = df
        self.num_centers = num_centers
        self.max_iter = max_iter
        self.epochs = epochs
        self.stop_threshold = 0.01 * (max(max(self.df['x']),max(self.df['y'])) - min(min(self.df['x']),min(self.df['y'])))
        # stop process when the change in the cluster centers is less than 1% of the range of the data
        self.best_model_eval = 10e9
        # best model fitness over all epochs
        self.iter_plots = []
        # save plots for visualization
        self.model_fitness = []
        # save model fitness evaluations after every epoch
        self.TOOLS = ['pan', 'wheel_zoom', 'box_zoom', 'box_select', 'reset', 'save', 'hover', 'crosshair']
        # initiating tools that enhance the graphing experience
        self.palette = Category10[10]
        # initiating a default color scheme
        # ***directly supplying a list of RGB tuples doesn't work (use default or HEX)
        self.colormapper = CategoricalColorMapper(factors=list(range(self.num_centers)),
                                                  palette=[self.palette[i] for i in range(self.num_centers)])
        # mapping clusters to colors
        prev_model_eval = 10e9
        self.all_iter_plots = []
        # saves iter_plots for visualization

        for epoch in range(epochs):
            self.iter_plots = []
            centers_delta = []
            self.randomize(self.df.shape[0])
            # assign every datapoint to a random cluster
            prev_centers = np.zeros((self.num_centers,len(self.df.columns) - 1))

            self.p = figure(tools=self.TOOLS, title='Epoch {epoch}'.format(epoch=epoch + 1))
            self.p.circle('x', 'y', color={'field': 'cluster', 'transform': self.colormapper}, size=8,
                          alpha=0.5, source=ColumnDataSource(self.df))
            # plot all datapoints
            self.iter_plots.append(self.p)
            # save the initial plot with randomly assigned clusters
            for iter in range(max_iter):
                self.p = figure(tools=self.TOOLS)
                centers, centers_used = self.calculate_centers()
                # centers_used is saved in case a center becomes unnecessary
                for idx, center in enumerate(centers):
                    self.p.circle(center[0], center[1], fill_color=self.palette[idx], line_color='black', size=15)
                    # plot all centers
                centers_dict = {idx: center for idx, center in enumerate(centers)}
                centers_delta.append(self.calculate_distance(centers, prev_centers[centers_used]))
                # save the center delta (masking over 'prev_centers' as necessary)
                if centers_delta[-1] < self.stop_threshold:
                    break
                # break out of iteration if delta is less than the threshold
                for index, row in self.df.iterrows():
                    self.reassign(index, [row['x'],row['y']], centers_dict)
                # assign every datapoint to the cluster of the closest cluster center
                prev_centers = np.array(centers)

                self.p.circle('x', 'y', color={'field': 'cluster', 'transform': self.colormapper}, size=8,
                              alpha = 0.5, source=ColumnDataSource(self.df))
                # plot reassigned datapoints on a new plot
                model_eval = self.evaluate_model(centers_dict)
                # evaluate the fitness of the model using the total within cluster sum of squared error metric
                self.p.title.text = 'Epoch {epoch}, Iteration {iter}   -   Model Fitness: {eval:.2f}, Delta of cluster centers: {delta:.2f}'.format(
                                    epoch=epoch + 1,iter=iter + 1, eval=model_eval, delta=centers_delta[-1])
                self.iter_plots.append(self.p)
                # save the plot
            self.model_fitness.append(model_eval)
            # save the model fitness evaluation
            if self.model_fitness[-1] < prev_model_eval:
                #*** duplicate edit variable error caused by trying to display the same plot twice
                #solution - rebuild best plot with different name
                self.best_model = figure(tools=self.TOOLS)
                for idx, center in enumerate(centers):
                    self.best_model.circle(center[0], center[1], fill_color=self.palette[idx], line_color='black', size=15)
                self.best_model.circle('x', 'y', color={'field': 'cluster', 'transform': self.colormapper}, size=8,
                                       alpha = 0.5, source=ColumnDataSource(self.df))
                self.best_model_eval = self.evaluate_model(centers_dict)
                self.best_model.title.text = 'Epoch {epoch}   -   Model Fitness: {eval:.2f}'.format(
                                              epoch=epoch + 1, eval=self.best_model_eval)

            prev_model_eval = self.model_fitness[-1]
            self.all_iter_plots.append(self.iter_plots)

    def randomize(self, n):
        ''' initial random assignment of clusters to observations '''
        self.df['cluster'] = np.random.randint(0, self.num_centers, n)

    def calculate_centers(self):
        ''' calculate the centers of every cluster '''
        centers = []
        centers_used = []
        for cluster in range(self.num_centers):
            center = np.mean(self.df[self.df['cluster'] == cluster]).tolist()[1:]
            if not pd.isnull(center).any():
                centers.append(center)
                centers_used.append(cluster)
        return centers, centers_used

    def reassign(self, index, row, centers_dict):
        ''' reassign the observation to the cluster of the closest center '''
        distances = [[index, self.calculate_distance(row, center)] for index, center in centers_dict.items()]
        closest_center = min(distances, key=lambda x: x[1])
        self.df.loc[index, 'cluster'] = closest_center[0]

    def calculate_distance(self, element1, element2):
        ''' calculate the Euclidean distance between two vectors '''
        return np.sqrt(np.sum((np.array(element1)-np.array(element2))**2))

    def evaluate_model(self, centers_dict):
        ''' evaluate the fitness of the model using the 'total within cluster sum of squared error' metric'''
        return sum([sum([self.calculate_distance([element[1]['x'], element[1]['y']], centers_dict[center])**2
               for element in self.df[self.df['cluster'] == center].iterrows()]) for center in range(len(centers_dict))])

def create_cluster(df, xs, ys, cluster):
    add_df = pd.DataFrame({'x':xs,'y':ys,'cluster':cluster})
    return df.append(add_df, ignore_index=True)

if __name__ == '__main__':
    #np.random.seed(1223)
    ''' user-specified values '''
    max_iter = 10
    epochs = 10

    df = pd.DataFrame(columns=['x', 'y', 'cluster'])
    ### specify characteristics of the data (i.e. mean and std of x and y coordinates, and number of datapoints)
    data_char = [{'x':[10,5], 'y':[10,5], 'n':20}, {'x':[50,5], 'y':[10,5], 'n':30},
                {'x':[10,5], 'y':[50,5], 'n':30}, {'x':[50,5], 'y':[50,5], 'n':30},
                {'x':[30,5], 'y':[30,5], 'n':50}]

    ### generate clusters of data (using the mean and median of the x and y values)
    for idx, datapoint in enumerate(data_char):
        df = create_cluster(df, xs=np.random.normal(*datapoint['x'], datapoint['n']),
                            ys=np.random.normal(*datapoint['y'], datapoint['n']), cluster=idx)

    # iterate through 1 to n values for k
    i = 0
    prev_model_eval = 10e9
    model_eval_delta = [-prev_model_eval]
    models = {}
    while True:
        i += 1
        classifier = kmeans(df.copy(deep=True), max_iter = max_iter, epochs= epochs, num_centers= i)
        models[i] = classifier
        print('k-means with {i} cluster(s):\t model fitness = {eval:.2f}'.format(i=i,eval=classifier.best_model_eval))
        model_eval_delta.append(classifier.best_model_eval - prev_model_eval)
        prev_model_eval = classifier.best_model_eval
        ### stop when model improvement is less than 25% (revert to previous model)
        if model_eval_delta[i] > -models[i].best_model_eval * 0.25:
            print('Model evaluation deltas: ', [float('{delta:.2f}'.format(delta=delta)) for delta in model_eval_delta])
            print('Optimal number of clusters (k): ', i - 1)
            break

    orig = figure()
    orig.circle('x', 'y', color={'field': 'cluster', 'transform': classifier.colormapper}, size=8,
                alpha = 0.5, source=ColumnDataSource(df))
    orig.title.text = '{num} clusters of data originally generated'.format(num=len(data_char))
    orig.circle([datapoint['x'][0] for datapoint in data_char], [datapoint['y'][0] for datapoint in data_char],
                color = 'black', size = 10)
    # plot original data cluster membership along with optimal centers in black

    models[i - 1].best_model.circle([datapoint['x'][0] for datapoint in data_char], [datapoint['y'][0] for datapoint in data_char],
                                     color = 'black', size = 10)
    # plot optimal centers in black in the visualization of the best model found

    for iter_plots in models[i-1].all_iter_plots:
        iter_plots[-1].circle([datapoint['x'][0] for datapoint in data_char], [datapoint['y'][0] for datapoint in data_char],
                               color = 'black', size = 10)
    # plot optimal centers in black in the visualization of the last model in each iteration

    iterations = Panel(child=Row(*[Column(*[plot for plot in iter_plots]) for iter_plots in models[i-1].all_iter_plots]),
                       title='Epochs and Iterations')
    # display a row of columns in one tab (iterations of the optimal k-means model)
    original = Panel(child=orig, title='Original Cluster Membership')
    # display the original plot in another tab
    best = Panel(child=models[i-1].best_model, title='Best Model Found')
    # display the best model found in the last tab
    tabs = Tabs(tabs=[original, iterations, best])

    output_file('k-means_output.html', title='k-means with dynamic k')
    show(tabs)
