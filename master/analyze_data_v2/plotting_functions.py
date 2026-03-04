
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import verify_data as verify

#compute and optionally plot a peri-stimulus time histogram
#plot is a line plot, with options for error display [bars or shaded]
def psth_line(times,triggers,pre=0.5,timeDomain=True,post=1,binsize=0.05,ymax=75,yoffset=0,output='fig',name='',color='#00cc00',linewidth=0.5,axes=None,labels=True,sparse=False,labelsize=18,axis_labelsize=20,error='shaded',alpha=0.5,**kwargs):
    post = post + 1
    peris=[]#np.zeros(len(triggers),len(times))
    p=[]
    if timeDomain:
        samplingRate = 1.0
    else:
        samplingRate = samplingRate
        
    times = np.array(times).astype(float) / samplingRate + pre
    triggers = np.array(triggers).astype(float) / samplingRate

    numbins = int((post+pre) / binsize) 
    bytrial = np.zeros((len(triggers),numbins))
    for i,t in enumerate(triggers):
        
        if len(np.where(times >= t - pre)[0]) > 0 and len(np.where(times >= t + post)[0]) > 0:
            start = np.where(times >= t - pre)[0][0]
            end = np.where(times >= t + post)[0][0]
            for trial_spike in times[start:end-1]:
                if float(trial_spike-t)/float(binsize) < float(numbins):
                    bytrial[i][int((trial_spike-t)/binsize-1)] +=1   
        else:
            pass
             #bytrial[i][:]=0
        #print 'start: ' + str(start)+'   end: ' + str(end)

    variance = np.std(bytrial,axis=0)/binsize/np.sqrt((len(triggers)))
    hist = np.mean(bytrial,axis=0)/binsize
    edges = np.linspace(-pre+binsize,post+binsize,numbins)

    if output == 'fig':
        if error == 'shaded':
            if 'shade_color' in kwargs.keys():
                shade_color=kwargs['shade_color']
            else:
                shade_color=color    
            if axes == None:
                plt.figure()
                axes=plt.gca()
            plt.locator_params(axis='y',nbins=4)
            upper = hist+variance
            lower = hist-variance
            axes.fill_between(edges[2:-1],upper[2:-1]+yoffset,hist[2:-1]+yoffset,alpha=alpha,color=shade_color,facecolor=shade_color)
            axes.fill_between(edges[2:-1],hist[2:-1]+yoffset,lower[2:-1]+yoffset,alpha=alpha,color=shade_color,facecolor=shade_color)
            axes.plot(edges[2:-1],hist[2:-1]+yoffset,color=color,linewidth=linewidth)
            axes.set_xlim(-pre,post-1)
            axes.set_ylim(0,ymax);
            if sparse:
                axes.set_xticklabels([])
                axes.set_yticklabels([])
            else:
                if labels:
                    axes.set_xlabel(r'$time \/ [s]$',fontsize=axis_labelsize)
                    axes.set_ylabel(r'$firing \/ rate \/ [Hz]$',fontsize=axis_labelsize)
                    axes.tick_params(axis='both',labelsize=labelsize)
            axes.spines['top'].set_visible(False);axes.yaxis.set_ticks_position('left')
            axes.spines['right'].set_visible(False);axes.xaxis.set_ticks_position('bottom')   
            axes.set_title(name,y=1)
            return axes 
        else:
            if axes == None:
                plt.figure()
                axes=plt.gca()
            f=axes.errorbar(edges,hist,yerr=variance,color=color)
            axes.set_xlim(-pre,post - 1)
            axes.set_ylim(0,ymax)
            if sparse:
                axes.set_xticklabels([])
                axes.set_yticklabels([])
            else:
                if labels:
                    axes.set_xlabel(r'$time \/ [s]$',fontsize=axis_labelsize)
                    axes.set_ylabel(r'$firing \/ rate \/ [Hz]$',fontsize=axis_labelsize)
                    axes.tick_params(axis='both',labelsize=labelsize)
            axes.spines['top'].set_visible(False);axes.yaxis.set_ticks_position('left')
            axes.spines['right'].set_visible(False);axes.xaxis.set_ticks_position('bottom')   
            axes.set_title(name)
            return axes
    if output == 'hist':
        return (hist[:-1*int(1./binsize)],edges[:-1*int(1./binsize)])    
    if output == 'p':
        return (edges,hist,variance)
	
#find the x and y position of an average waveform, given the probe geometry
#plot is a bar plot
def psth(times,triggers,timeDomain=False,pre=0.5,post=1,binsize=0.05,ymax=75,output='fig',name='',color=[1,1,1],axes=None,labels=True,sparse=False,labelsize=18):
    peris=[]#np.zeros(len(triggers),len(times))
    if timeDomain:
        samplingRate = 1.0
    else:
        samplingRate = samplingRate
        
    times = np.array(times).astype(float) / samplingRate
    triggers = np.array(triggers).astype(float) / samplingRate
    
    for i,t in enumerate(triggers):
        peris.append(np.array(times).astype(float)-float(t))
    peris = np.array(peris)
    peris=peris.flatten()

    numbins = (post+pre) / binsize 
    (hist,edges) = np.histogram(peris,int(numbins),(-pre,post))
    hist /= float(len(triggers)*binsize)


    if output == 'fig':
        if axes == None:
            plt.figure()
            axes=plt.gca()
        f=axes.bar(edges[:-1],hist,width=binsize,color=color)
        axes.set_xlim(-pre,post)
        axes.set_ylim(0,ymax)
        if sparse:
            axes.set_xticklabels([])
            axes.set_yticklabels([])
        else:
            if labels:
                axes.set_xlabel(r'$time \/ [s]$',fontsize=20)
                axes.set_ylabel(r'$firing \/ rate \/ [Hz]$',fontsize=20)
                axes.tick_params(axis='both',labelsize=labelsize)
        axes.spines['top'].set_visible(False);axes.yaxis.set_ticks_position('left')
        axes.spines['right'].set_visible(False);axes.xaxis.set_ticks_position('bottom')
        axes.set_title(name)
        return axes
    if output == 'hist':
        return (hist,edges)    
    if output == 'p':
        return peris

# def trial_by_trial(spike_times, event_times, pre, post, bin_size):
#     if bin_size > pre+post:
#         print('bin_size larger than window')
#         return np.nan,np.nan,np.nan,np.nan
#     if int((pre+post)*1e4) % int(bin_size*1e4) > 0:
#         print('bin_size doesn\'t evenly fit into window')
#         return np.nan,np.nan,np.nan,np.nan
    
#     spike_times = np.array(spike_times).astype(float) + pre
#     event_times = np.array(event_times).astype(float)

#     numbins  = np.round(np.ceil((pre+post)/bin_size)).astype(int)
#     edges    = np.linspace(-pre,post,numbins)
    
#     bytrial  = np.zeros((len(event_times),numbins))
#     var      = np.zeros((numbins))
#     psth     = np.zeros((numbins))
    

#     for t,time in enumerate(event_times):
#         if len(np.where(spike_times >= time - pre)[0]) > 0 and len(np.where(spike_times >= time + post)[0]) > 0:
#             start = np.where(spike_times >= (time - pre))[0][0]
#             end   = np.where(spike_times >= (time + post))[0][0]
                
#             for trial_spike in spike_times[start:end]:
#                 b = np.round(np.ceil((trial_spike-time)/bin_size)).astype(int) - 1 #same way we calculate numbins
#                 if b <= numbins:
#                     bytrial[t,:][b] +=1
#         else:
#             continue

#     var  = np.nanstd(bytrial,axis=0)/bin_size/np.sqrt(len(event_times))
#     psth = np.nanmean(bytrial,axis=0)/bin_size

#     return psth, var, edges, bytrial

def raster(times,triggers,pre=0.5,timeDomain=False,post=1,yoffset=0,output='fig',name='',color='#00cc00',linewidth=0.5,axes=None,labels=True,sparse=False,labelsize=18,axis_labelsize=20,error='',alpha=0.5,ms=2,**kwargs):
    #post = post + 1
    if timeDomain:
        samplingRate = 1.0
    else:
        samplingRate = samplingRate
        
    times = np.array(times).astype(float) / samplingRate + pre
    triggers = np.array(triggers).astype(float) / samplingRate
    bytrial = []
    
    if axes == None and output!='data':
        plt.figure()
        axes=plt.gca()
        
    for i,t in enumerate(triggers):
        if len(np.where(times >= t - pre - post)[0]) > 0 and len(np.where(times >= t + post+ pre)[0]) > 0:
            start = np.where(times >= t - pre)[0][0]
            end = np.where(times >= t + post)[0][0]
            bytrial.append(np.array(times[start:end])-t-pre)
            if output!='data':
            #		print np.ones(len(np.array(times[start:end-1])-t))*i+1
                axes.plot(np.array(times[start:end])-t-pre,
                          np.ones(len(np.array(times[start:end])-t))*i+1,
                          "|",mew=linewidth,ms=ms,color=color)
        else: bytrial.append([])
                
                
    if output!='data':
        axes.set_xlim(-pre,post)
        axes.set_title(name)
        axes.set_ylim(len(triggers),1)
        if sparse:
            cleanAxes(axes,total=True)
        else:
            if labels:
                axes.set_xlabel(r'$time \/ [s]$',fontsize=16)
                axes.set_ylabel(r'$trial \/ number$',fontsize=16)
                axes.tick_params(axis='both',labelsize=labelsize)
                axes.spines['top'].set_visible(False);axes.yaxis.set_ticks_position('left')
                axes.spines['right'].set_visible(False);axes.xaxis.set_ticks_position('bottom')
         

    if output == 'fig':
        return (plt.gcf(),plt.gca())
    if output=='data':
        return bytrial
         


def raster_singletrial(nwb_data,trigger,pre=0.5,timeDomain=True,post=.1,insertion_angle=45,yoffset=0,output='fig',name='',color='#00cc00',linewidth=0.5,axes=None,labels=True,sparse=False,labelsize=18,axis_labelsize=20,error='',alpha=0.5,**kwargs):
	post = post + 1
	if timeDomain:
		samplingRate = 1.0
	else:
		samplingRate = samplingRate
        
	t = float(trigger) / samplingRate
	bycell = [];
	
	if axes == None:
		plt.figure()
		axes=plt.gca()
	for ii,probe in enumerate(nwb_data['processing'].keys()):
		if 'UnitTimes' in nwb_data['processing'][probe]:
			for i,cell in enumerate(np.sort(nwb_data['processing'][probe]['UnitTimes'].keys())[:-1]):
				times = nwb_data['processing'][probe]['UnitTimes'][cell]['times']
				if len(np.where(times >= t - pre)[0]) > 0 and len(np.where(times >= t + post)[0]) > 0:
					start = np.where(times >= t - pre)[0][0]
					end = np.where(times >= t + post)[0][0]
					bycell.append(np.array(times[start:end-1])-t)
					axes.plot(np.array(times[start:end-1])-t,
							  np.ones(len(np.array(times[start:end-1])-t))*(nwb_data['processing'][probe]['UnitTimes'][cell]['ypos']*np.sin(np.deg2rad(90-insertion_angle))),
							  '|',
							  linewidth=1,mew=0.5,
							  #color='#d9d9d9')
							  color=color50[ii%50])
	axes.set_xlim(-pre,post-1.)
	axes.set_ylim(1000,0)
	axes.set_title(name)

	
	if sparse:
		cleanAxes(axes,total=True)
	else:
		if labels:
			axes.set_xlabel(r'$time \/ [s]$',fontsize=16)
			axes.set_ylabel(r'$depth \/ um$',fontsize=16)
			axes.tick_params(axis='both',labelsize=labelsize)
			axes.spines['top'].set_visible(False);axes.yaxis.set_ticks_position('left')
			axes.spines['right'].set_visible(False);axes.xaxis.set_ticks_position('bottom')
	if output == 'fig':
		return (plt.gcf(),plt.gca())
	if output=='data':
		return bycell
        
#compute the tuning over a given parameter from a PSTH
def psth_tuning(data,unit,param,params,paramtimesdict,window=1.33,binsize=0.02,savepsth=False,path=r'C:\Users\danield\Desktop\data'):
    tun_y = np.zeros(len(params))
    tun_x = np.zeros(len(params))
    i=0
    for p in params:
        print(p)
        if savepsth:
            f=psth(data[unit]['times'],paramtimesdict[param+str(p)],pre=0,post=window,binsize=binsize)
            f.savefig(os.path.join(path,'unit'+unit+param+str(p)+'_psth.eps'),format='eps')     
        
        (hist,edges) = psth(data[unit]['times'],paramtimesdict[param+str(p)],pre=0,post=window,binsize=binsize,output='hist')        
        tun_y[i] = np.mean(hist)
        tun_x[i] = p
        i+=1
        
    plt.plot(tun_x,tun_y,'ko-');
    plt.xscale('log');
    plt.xlim(7,101);
    plt.ylabel(r'$firing \/ rate \/ [Hz]$',fontsize=14);
    plt.xlabel(r'$contrast \/ $[%]',fontsize=14);
    f=plt.gcf()
    return f

#compute the latency to first reseponse from a PSTH
def psth_latency(data,bins,pre=None,binsize=None, sd = 2.5,smooth=False,offset=0):
    if smooth:
        data = scipy.signal.savgol_filter(data,5,3)
    if pre is None:
        pre = bins[0]
    if binsize == None:
        binsize = bins[1]-bins[0]
    startbin = np.where(bins>0)[0][0]
    baseline = np.mean(data[:startbin])
    threshold = baseline + np.std(data[:startbin])*sd +0.2
    crossings = plt.mlab.cross_from_below(data[startbin:],threshold)
    if len(crossings)>0:
        crossing = crossings[0]#the first bin above the threshold
        chunk = np.linspace(data[crossing+startbin-1],data[crossing+startbin],100)
        bin_crossing = plt.mlab.cross_from_below(chunk,threshold)
        latency =(crossing-1)*(1000*binsize)+bin_crossing/100.0 * (1000*binsize) 
    else:
        #print 'response did not exceed threshold: '+str(threshold)+', no latency returned'
        return None
    return latency[0] - offset

#compute the latency to first reseponse from a PSTH
def psth_area(data,bins,pre=None,binsize=None, sd = 3,time=0.2):
    if pre is None:
        pre = bins[0]
    if binsize == None:
        binsize = bins[1]-bins[0]
    startbin = np.where(bins>0)[0][0]
    baseline = np.mean(data[:startbin])
    threshold = baseline + np.std(data[:startbin])*sd +0.2
    crossings = plt.mlab.cross_from_below(data[startbin:],threshold)
    if len(crossings)>0:
        try:
            area = np.trapz(np.abs(data[startbin:startbin+np.ceil(time/binsize)]) - baseline)
            return area
        except: return None 
        print('response did not exceed threshold: '+str(threshold)+', no area returned')
        return None

def psth_line_overlay_(spike_data, unit, stim_data, condition, title='', 
                       pre=0.5, post=2.5,binsize=0.05,variance=True,axis=None,legend=True):
#     times = np.array(np.array(spike_data.times[spike_data.unit_id==unit])[0])
    times = np.array(spike_data[spike_data.unit_id==unit].times.values[0])
    numbins = int((post+pre)/binsize)
    conds = np.unique(stim_data[condition])
    num_conds = len(conds)
    x = np.arange(-pre,post,binsize)
    colors = plt.cm.viridis(np.linspace(0,1,num_conds))

    psth_all=[]

    if axis == None:
        fig,ax = plt.subplots()
    else:
        ax = axis; fig = plt.gcf()

    for i,cond in enumerate(np.unique(stim_data[condition])):
        triggers = np.array(stim_data['times'][stim_data[condition] == cond])
        bytrial = np.zeros((len(triggers),numbins-1))
        for j, trigger in enumerate(triggers):
            trial = triggers[j]
            start = trial-pre
            end = trial+post
            bins_ = np.arange(start,end,binsize)
            trial_spikes = times[np.logical_and(times>=start, times<=end)]
            hist,edges = np.histogram(trial_spikes,bins=bins_)
            if len(hist)==numbins-1:
                bytrial[j]=hist
            elif len(hist)==numbins:
                bytrial[j]=hist[:-1]
        psth = np.mean(bytrial,axis=0)/binsize
        if isinstance(conds[i],float)==True:
            ax.plot(x[:-1],psth, color=colors[i], label=str(round(conds[i],2)))
#         if isinstance(conds[i],tuple)==True:
#             ax.plot(x[:-1],psth, color=colors[i], label=str(round(conds[i],2)))
        else:
            ax.plot(x[:-1],psth, color=colors[i], label=str(conds[i]))
        if variance == True:
            var = np.std(bytrial,axis=0)/binsize/np.sqrt((len(triggers)))
            upper = psth+var
            lower = psth-var
            ax.fill_between(x[:-1],upper,psth,alpha=0.1,color=colors[i])
            ax.fill_between(x[:-1],lower,psth,alpha=0.1,color=colors[i])
    ax.axvline(0,linestyle='dashed')
    if legend==True:
        plt.legend(loc=(1.05,0.48))
    plt.title(title)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
#     return(ax)

def trial_by_trial(spike_times, event_times, pre, post, bin_size,brain_region=None):

    if bin_size > pre+post:
        print('bin_size larger than window')
        return np.nan,np.nan,np.nan,np.nan
    if int((pre+post)*1e4) % int(bin_size*1e4) > 0:
        print('bin_size doesnt evenly fit into window')
        return np.nan,np.nan,np.nan,np.nan
    
    # spike_times = np.array(spike_times).astype(float) + pre
    spike_times = np.array(spike_times).astype(float) 
    event_times = np.array(event_times).astype(float)

    numbins  = np.round(((pre+post)/bin_size)).astype(int)
    edges    = np.linspace(-pre,post,numbins)
    

    bytrial  = np.zeros((len(event_times),numbins))
    var      = np.zeros((numbins))
    psth     = np.zeros((numbins))
  
    for t,time in enumerate(event_times):
      
        if len(np.where(spike_times >= time - pre)[0]) > 0 and len(np.where(spike_times >= time + post)[0]) > 0:
            start = np.where(spike_times >= (time - pre))[0][0]
            end   = np.where(spike_times >= (time + post))[0][0]
           
    
            for trial_spike in spike_times[start:end]:
                # b = np.round(np.ceil((trial_spike-time)/bin_size)).astype(int) - 1 #same way we calculate numbins
                b = np.floor((trial_spike - time + pre) / bin_size).astype(int)

                
                if b <= numbins:
                    bytrial[t,:][b] +=1
                   
        else:
            continue

    var  = np.nanstd(bytrial,axis=0)/bin_size/np.sqrt(len(event_times))
    psth = np.nanmean(bytrial,axis=0)/bin_size

    return psth, var, edges, bytrial

def trial_by_trial_subplots(spike_times, event_times, pre, post, bin_size):

    if bin_size > pre+post:
        print('bin_size larger than window')
        return np.nan,np.nan,np.nan,np.nan
    if int((pre+post)*1e4) % int(bin_size*1e4) > 0:
        print('bin_size doesnt evenly fit into window')
        return np.nan,np.nan,np.nan,np.nan
    
    # spike_times = np.array(spike_times).astype(float) + pre
    spike_times = np.array(spike_times).astype(float) 
    event_times = np.array(event_times).astype(float)

    numbins  = np.round(((pre+post)/bin_size)).astype(int)
    edges    = np.linspace(-pre,post,numbins)
    

    bytrial  = np.zeros((len(event_times),numbins))
    var      = np.zeros((numbins))
    psth     = np.zeros((numbins))
  
    for t,time in enumerate(event_times):
      
        if len(np.where(spike_times >= time - pre)[0]) > 0 and len(np.where(spike_times >= time + post)[0]) > 0:
            start = np.where(spike_times >= (time - pre))[0][0]
            end   = np.where(spike_times >= (time + post))[0][0]
           
    
            for trial_spike in spike_times[start:end]:
                # b = np.round(np.ceil((trial_spike-time)/bin_size)).astype(int) - 1 #same way we calculate numbins
                b = np.floor((trial_spike - time + pre) / bin_size).astype(int)

                
                if b <= numbins:
                    bytrial[t,:][b] +=1
                   
        else:
            continue

    var  = np.nanstd(bytrial,axis=0)/bin_size/np.sqrt(len(event_times))
    psth = np.nanmean(bytrial,axis=0)/bin_size

    return psth, var, edges, bytrial

def singleUnit_psth_raster_test(spike_times, event_times,unit_id, pre, post, bin_size=0.01):
    fig, ax = plt.subplots(2,1)  # Initialize the plot
    fig.set_size_inches(10, 8)  # Set the figure size

    psth,var,edges,bytrial = trial_by_trial(spike_times, event_times, pre, post, bin_size)

    mean_baseline = bytrial[:,:int(pre/bin_size)].mean()
    std_baseline  = bytrial[:,:int(pre/bin_size)].std()
    zMean         = (bytrial[:,int(pre/bin_size):].mean(axis=1) - mean_baseline)/std_baseline

    # mean_response = bytrial[:,int(pre/bin_size):].mean()
    # std_response  = bytrial[:,int(pre/bin_size):].std()

    n_bins = len(psth)
    ax[0].plot(edges,psth-mean_baseline,color='b')
    # ax.set_xticks(np.linspace(0,n_bins,6),np.round(np.linspace(-pre,post,6),1))
    # ax.fill_between(np.arange(0,n_bins),psth+var-mean_baseline,psth-var-mean_baseline,alpha=0.2,color='b')
    ax[0].fill_between(edges,psth+var-mean_baseline,psth-var-mean_baseline,alpha=0.2,color='b')

    ax[0].set_ylabel('Firing Rate [Hz]')
    ax[0].set_xlabel('Time [s]')
    ax[0].set_title(f'PSTH - {unit_id}')


    for t, time in enumerate(event_times):
        trial_spikes = spike_times[(spike_times > time-pre) & (spike_times < time+post)]
        trial_spikes = trial_spikes - time
        # print(f'trial_spikes: {trial_spikes[0:5]}')
        # print(f'time-post: {time-post}')
        # print(f'time+post: {time+post}')

        ax[1].scatter(trial_spikes,
                [t] * len(trial_spikes),  # Use `t` as the y-value for each trial
                marker='|',
                color='black',
                s=1,
                alpha=0.6)

    return ax
