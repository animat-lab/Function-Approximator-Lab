#!/usr/local/bin/python

"""
__author__ = "Ryan Lober"
__credits__ = "Olivier Sigaud, Nicolas Perrin, Didier Marin and Florian Lesaint"
__version__ = "1.0"
"""


import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def plotFA(fa_object, data):
	"""
	This function can be used to plot all of the relevant approximator data and performance statistic, as well as an overlay with the original function."
	
	:param fa_object: a trained function approximator object
	:param data: original noisy data set
	"""
	numFeats = fa_object.numFeatures
	
	# Reconstruct the original function
	Np = np.shape(data)[1]
	x_values = data[0,:]
	y_noisy = data[1,:]
	y_values = 1 - x_values - np.sin(-2.*math.pi*np.power(x_values,3.))*np.cos(-2.*math.pi*np.power(x_values,3.))*np.exp(-np.power(x_values,4.))

	# Get the output of each of the features
	if fa_object.method == 'LWLS':
		phi = fa_object.featureOutput(x_values)
		y_feats = (np.dot(phi.transpose(), fa_object.theta)).transpose() #[numFeats x Ns]
		y_featWeights = fa_object.getWeights(x_values)

	else:
		y_feats = fa_object.featureOutput(x_values)

	# Get the predicted values from the FA
	y_approx = fa_object.functionApproximatorOutput(x_values)

	
	# Get execution time info
	totalTime = 0.0
	timeline = [totalTime]
	for t in fa_object.timeHistory:
		totalTime += t
		timeline.append(totalTime)
	
	trainingTime = timeline[-1:][0]
	iterTime = np.mean(fa_object.timeHistory)
	
	# Get total error
	totalError = np.sum(np.abs(y_values - y_approx))
	
	
	### Plots
	############################################################################
	
	# Set up various subplots
	plt.figure(num=1, figsize=(16, 9), dpi=80, facecolor='w', edgecolor='k')
	ax_main = plt.subplot2grid((3,3), (0,0), colspan=2, rowspan=3)
	ax_cost = plt.subplot2grid((3,3), (0,2))
	ax_cpu = plt.subplot2grid((3,3), (1,2))
	ax_mem = plt.subplot2grid((3,3), (2,2))

	# Main plot with data and FA 
	f_real, = ax_main.plot(x_values, y_values)
	f_noisy, = ax_main.plot(x_values, y_noisy, '*')
	f_approx, = ax_main.plot(x_values, y_approx, 'r')
	for i in range(numFeats):
		if fa_object.method == 'LWLS':
			for j in range(Np):
				feature_plots, = ax_main.plot(x_values[j], y_feats[i,j], '_', mec='k', alpha=y_featWeights[i,j], mew=2)
		else:
			feature_plots, = ax_main.plot(x_values, y_feats[i,:]*fa_object.theta[i], 'k--')

	
	
	
	ax_main.legend((f_real, f_noisy, f_approx, feature_plots), (r'$f_{real}$', r'$y_{noisy}$', r'$\hat{f}$',r'$\phi * \Theta$'))
	ax_main.set_title('Function approximator output')
	ax_main.set_xlabel('input')
	ax_main.set_ylabel('output')
	ax_main.set_ylim([-1., 1.5])
	
	statString = 'FA Error: '+str(round(totalError*1000)/1000)+'\nTraining Time: '+str(round(trainingTime*1E6)/1E3)+' ms\nMean Iteration Time: ' + str(round(iterTime*1E6)/1E3) +' ms'
	#ax_main.annotate(statString, xy=(0.1, 0.05), textcoords='axes fraction')
	bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.7)
	ax_main.text(0.05, -.9, statString, ha="left", va="bottom", size=12,
        bbox=bbox_props)

	# Cost curve (if using an iterative method)
	if fa_object.iterationCount > 1 and np.size(fa_object.deltaHistory) > 1: 
		ax_cost.plot(range(fa_object.iterationCount), fa_object.deltaHistory)
		ax_cost.axhline(y = fa_object.threshold, c='red', ls='--', lw=3)
	
	ft = 9
	ax_cost.set_xlabel('Iteration', fontsize=ft)
	ax_cost.set_ylabel('Cost', fontsize=ft)
	ax_cost.set_ylim([0,0.01])
	ax_cost.tick_params(axis='both', which='major', labelsize=ft)
	
	
	
	
	
	cpu_labels = []
	ax_cpu.plot(timeline, fa_object.cpuHistory)
	for cpu in range(np.size(fa_object.cpuHistory[0])):
		cpu_labels.append('core_'+str(cpu))
	

	ax_cpu.legend(cpu_labels, bbox_to_anchor=(1.3, 1.00))
	ax_cpu.set_xlabel('Time (sec)', fontsize=ft)
	ax_cpu.set_ylabel('CPU usage (%)', fontsize=ft)
	ax_cpu.tick_params(axis='both', which='major', labelsize=ft)
	ax_cpu.set_ylim([-10, 110])

	
	ax_mem.plot(timeline, fa_object.memHistory)
	ax_mem.set_xlabel('Time (sec)', fontsize=ft)
	ax_mem.set_ylabel('RAM usage (%)', fontsize=ft)
	ax_mem.tick_params(axis='both', which='major', labelsize=ft)
	
	plt.subplots_adjust(wspace=0.25, hspace=0.25)
	plt.show()
	
	# if fa_object.method == 'Incremental':
# 		animFAPlot(fa_object, data)
		


		
def animPlotFA(fa_object, data):
	"""
	For iterative methods, this function can be used to plot the evolution of the approximator over its training."
	
	:param fa_object: a trained function approximator object
	:param data: original noisy data set
	"""


	assert(fa_object.method == 'Incremental')
	Np = np.shape(data)[1]
	x_values = data[0,:]
	y_noisy = data[1,:]
	y_values = 1 - x_values - np.sin(-2.*math.pi*np.power(x_values,3.))*np.cos(-2.*math.pi*np.power(x_values,3.))*np.exp(-np.power(x_values,4.))
	numFeats = fa_object.numFeatures
	
	fig2 = plt.figure()
	ax = fig2.add_subplot(1, 1, 1)
	#num=2, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
	f_r, = ax.plot(x_values, y_values)
	f_n, = ax.plot(x_values, y_noisy, '*')
	y_iter = fa_object.functionApproximatorOutput(x_values, fa_object.thetaHistory[0])
	f_app, = ax.plot(x_values, y_iter, 'r')
	y_feats = fa_object.featureOutput(x_values)
	ax.set_title('Function approximator output')
	ax.set_xlabel('input')
	ax.set_ylabel('output')
	ax.set_ylim([-1., 1.5])
	

		
	def animate(j, fa_object, x_values):
		y_it = fa_object.functionApproximatorOutput(x_values, fa_object.thetaHistory[j])
		f_app.set_ydata(y_it)  # update the data
		 
		# for i in range(numFeats):
#  			feature_plots, = ax.plot(x_values, y_feats[i,:]*fa_object.thetaHistory[j][i], 'k--')
 		statString = 'Iteration: '+str(j)
 		bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.7)
 		ax.text(0.05, -.9, statString, ha="left", va="bottom", size=12, bbox=bbox_props)
		return f_app, #feature_plots


	
	Nt = len(fa_object.deltaHistory)
	Nsteps = int(round(Nt/100))
	
	ani = animation.FuncAnimation(fig2, animate, np.arange(0,Nt,Nsteps), fargs=(fa_object, x_values),
		interval=1, blit=False, repeat=True)
	plt.show()

