"""
Script... 




Other color schemes that might work taken from http://colorbrewer2.org/ with
the chosen options of diverging/sequential and that are colorblind safe:

[["#2166ac", "#f7f7f7", "#b2182b"], 
["#4575b4", "#ffffbf", "#d73027"], 
["#542788", "#f7f7f7", "#b35806"], 
["#1b7837", "#f7f7f7", "#762a83"], 
["#4d9221", "#f7f7f7", "#c51b7d"],
["#01665e", "#f5f5f5", "#8c510a"],
["#ffffb2", "#fd8d3c", "#b10026"],
["#ffffcc", "#41b6c4", "#0c2c84"]]



Notes
-------------

Model a has 99 simulations for each control measure
Model b doesn't have "ring culling"


Contact
-------------

W. Probert, 2015
wjp11@psu.edu

"""

import pandas as pd, os, numpy as np, matplotlib
from matplotlib import pyplot as plt

import matplotlib.lines as mlines,  matplotlib.patches as mpatches
import matplotlib.colors as mcol, matplotlib.cm as cm
from matplotlib.table import Table

def run_analysis():
    """
    
    """
    
    main_dir = os.path.join('/', 'Users', 'wjmprobert', 'Projects', \
        'objectives_matter', 'objectives_matter')
    
    a = pd.read_csv(os.path.join(main_dir, 'data', 'model_a.csv'))
    b = pd.read_csv(os.path.join(main_dir, 'data', 'model_b.csv'))
    c = pd.read_csv(os.path.join(main_dir, 'data', 'model_c.csv'))
    d = pd.read_csv(os.path.join(main_dir, 'data', 'model_d.csv'))
    e = pd.read_csv(os.path.join(main_dir, 'data', 'model_e.csv'))
    
    full = pd.concat([a, b, c, d, e], axis = 0)
    
    # Total number of animals culled
    culled_cols = ['cattle_culled', 'sheep_culled', 'cattlesheep_culled']
    full['total_culled'] = full[culled_cols].sum(axis = 1)
    
    # Cost (in millions of pounds)
    full['cost'] = 1e-6*(1000*full.cattle_culled + \
        100*full.sheep_culled + \
        500*full.cattlesheep_culled + \
        1*full.cattle_vacc)
    
    # Create vaccinate-to-kill actions
    
    v03_2k = full.loc[full.control == "v03"]
    
    culled_cols_2k = ['cattle_culled', 'sheep_culled', \
        'cattlesheep_culled','cattle_vacc']
    
    del v03_2k['total_culled'], v03_2k['cost']
    
    v03_2k['total_culled'] = v03_2k[culled_cols_2k].sum(axis = 1)
    v03_2k['cost'] = 1e-6*(1000*v03_2k.cattle_culled + \
        100*v03_2k.sheep_culled + \
        500*v03_2k.cattlesheep_culled + \
        1000*v03_2k.cattle_vacc)
    
    v03_2k = v03_2k.replace("v03", "v03_2kill")
    
    v10_2k = full.loc[full.control == "v10"]
    del v10_2k['total_culled'], v10_2k['cost']
    
    v10_2k['total_culled'] = v10_2k[culled_cols_2k].sum(axis = 1)
    v10_2k['cost'] = 1e-6*(1000*v10_2k.cattle_culled + \
        100*v10_2k.sheep_culled + \
        500*v10_2k.cattlesheep_culled + \
        1000*v10_2k.cattle_vacc)
    
    v10_2k = v10_2k.replace("v10", "v10_2kill")
    
    full = pd.concat([full, v03_2k, v10_2k])
    
    # Create and save table of results for each objective
    objectives = ["duration", "cost", "total_culled"]
    thresholds = [50., 20., 12500.]
    
    for obj, thresh in zip(objectives, thresholds):
        
        filepath = os.path.join(main_dir, "data", "tab_"+obj + ".csv")
        
        tab = create_table(full, obj, thresh)
        tab.to_csv(filepath, index = True)
    
    # Create the value and color tables
    cost = pd.read_csv(os.path.join(main_dir, 'data', 'tab_cost.csv'))
    duration = pd.read_csv(os.path.join(main_dir, 'data', 'tab_duration.csv'))
    culled = pd.read_csv(os.path.join(main_dir, 'data', 'tab_total_culled.csv'))
    
    value_tab_cost, col_tab_cost = manipulate_table(cost)
    value_tab_cull, col_tab_cull = manipulate_table(culled)
    value_tab_dur, col_tab_dur = manipulate_table(duration)
    
    # Make tables (with metrics as column headers)
    means = pd.concat([value_tab_cost.Mean, \
    value_tab_dur.Mean, \
    value_tab_cull.Mean], axis = 1, ignore_index = True)
    means.columns = ["Cost", "Duration", "Livestock culled"]
    
    means_col = pd.concat([col_tab_cost.Mean, \
    col_tab_dur.Mean, \
    col_tab_cull.Mean], axis = 1, ignore_index = True)
    means_col.columns = ["Cost", "Duration", "Livestock culled"]
    
    medians = pd.concat([value_tab_cost.Median, \
    value_tab_dur.Median, \
    value_tab_cull.Median], axis = 1, ignore_index = True)
    medians.columns = ["Cost", "Duration", "Livestock culled"]
    
    medians_col = pd.concat([col_tab_cost.Median, \
    col_tab_dur.Median, \
    col_tab_cull.Median], axis = 1, ignore_index = True)
    medians_col.columns = ["Cost", "Duration", "Livestock culled"]
    
    variances = pd.concat([value_tab_cost.Variance, \
    value_tab_dur.Variance, \
    value_tab_cull.Variance], axis = 1, ignore_index = True)
    variances.columns = ["Cost", "Duration", "Livestock culled"]
    
    variances_col = pd.concat([col_tab_cost.Variance, \
    col_tab_dur.Variance, \
    col_tab_cull.Variance], axis = 1, ignore_index = True)
    variances_col.columns = ["Cost", "Duration", "Livestock culled"]
    
    quantiles = pd.concat([value_tab_cost.Quantile90, \
    value_tab_dur.Quantile90, \
    value_tab_cull.Quantile90], axis = 1, ignore_index = True)
    quantiles.columns = ["Cost", "Duration", "Livestock culled"]
    
    quantiles_col = pd.concat([col_tab_cost.Quantile90, \
    col_tab_dur.Quantile90, \
    col_tab_cull.Quantile90], axis = 1, ignore_index = True)
    quantiles_col.columns = ["Cost", "Duration", "Livestock culled"]
    
    # Subset the duration objective to exclude vaccinate to kill/live
    
    value_tab_dur = value_tab_dur.loc[(['aip', 'dc', 'rc', 'v03', 'v10'], slice(None))]
    col_tab_dur = col_tab_dur.loc[(['aip', 'dc', 'rc', 'v03', 'v10'], slice(None))]
    
    # Make a subset of statistics to use
    subset_dur = ['Median', 'Quantile90', 'EmpProb50']
    
    # Make figures (with statistics as column headers)
    fig, ax = make_tab_dur(value_tab_dur, col_tab_dur)
    figurepath = os.path.join(main_dir, 'graphics', 'table_duration.pdf')
    plt.savefig(figurepath, dpi = 400)
    plt.close()
    
    # Make a subset plot
    fig, ax = make_tab_dur(value_tab_dur.loc[:, subset_dur], \
        col_tab_dur.loc[:, subset_dur])
    figurepath = os.path.join(main_dir, 'graphics', 'table_duration_mqe.pdf')
    plt.savefig(figurepath, dpi = 400)
    plt.close()
    
    # Make figures for each model
    idx = pd.IndexSlice
    for m in ['A', 'B', 'C', 'D', 'E']:
        ctrls = ['aip', 'dc', 'rc', 'v03', 'v10']
        moda_val = value_tab_dur.loc[idx[ctrls,[m]],:]
        moda_col = col_tab_dur.loc[idx[ctrls,[m]],:]
        fig, ax = make_tab_dur(moda_val, moda_col)
        figurepath = os.path.join(main_dir, 'graphics',\
            'table_duration_model_'+m+'.pdf')
        plt.savefig(figurepath, dpi = 400)
        plt.close()
        
        fig, ax = make_tab_dur(moda_val.loc[:,subset_dur], \
             moda_col.loc[:,subset_dur])
        figurepath = os.path.join(main_dir, 'graphics', \
            'table_duration_model_'+m+'_mqe.pdf')
        plt.savefig(figurepath, dpi = 400)
        plt.close()
    
    subset_cost = ['Median', 'Quantile90', 'EmpProb20']
    
    fig, ax = make_tab_big(value_tab_cost, col_tab_cost, format_type = "cost")
    figurepath = os.path.join(main_dir, 'graphics', 'table_cost.pdf')
    plt.savefig(figurepath, dpi = 400)
    plt.close()
    
    fig, ax = make_tab_big(value_tab_cost.loc[:,subset_cost], \
        col_tab_cost.loc[:,subset_cost], \
        format_type = "cost")
    figurepath = os.path.join(main_dir, 'graphics', 'table_cost_mqe.pdf')
    plt.savefig(figurepath, dpi = 400)
    plt.close()
    
    # Make figures for each model
    idx = pd.IndexSlice
    for m in ['A', 'B', 'C', 'D', 'E']:
        moda_val = value_tab_cost.loc[idx[:,[m]],:]
        moda_col = col_tab_cost.loc[idx[:,[m]],:]
        fig, ax = make_tab_big(moda_val, moda_col, format_type = "cost")
        figurepath = os.path.join(main_dir, 'graphics', 'table_cost_model_'+m+'.pdf')
        plt.savefig(figurepath, dpi = 400)
        plt.close()
        
        fig, ax = make_tab_big(moda_val.loc[:,subset_cost], \
            moda_col.loc[:,subset_cost], \
            format_type = "cost")
        figurepath = os.path.join(main_dir, 'graphics', 'table_cost_model_'+m+'_mqe.pdf')
        plt.savefig(figurepath, dpi = 400)
        plt.close()
    
    subset_culled = ['Median', 'Quantile90', 'EmpProb12500']
    
    # Culled table
    fig, ax = make_tab_big(value_tab_cull, col_tab_cull, format_type = "culled")
    figurepath = os.path.join(main_dir, 'graphics', 'table_culled.pdf')
    plt.savefig(figurepath, dpi = 400)
    plt.close()
    
    # Culled table using a subset of statistics
    fig, ax = make_tab_big(value_tab_cull.loc[:,subset_culled], \
        col_tab_cull.loc[:,subset_culled], \
        format_type = "culled")
    figurepath = os.path.join(main_dir, 'graphics', 'table_culled_mqe.pdf')
    plt.savefig(figurepath, dpi = 400)
    plt.close()
    
    # Make figures for each model
    idx = pd.IndexSlice
    for m in ['A', 'B', 'C', 'D', 'E']:
        moda_val = value_tab_cull.loc[idx[:,[m]],:]
        moda_col = col_tab_cull.loc[idx[:,[m]],:]
        fig, ax = make_tab_big(moda_val, moda_col, format_type = "culled")
        figurepath = os.path.join(main_dir, 'graphics', 'table_culled_model_'+m+'.pdf')
        plt.savefig(figurepath, dpi = 400)
        plt.close()
        
        fig, ax = make_tab_big(moda_val.loc[:,subset_culled], \
            moda_col.loc[:,subset_culled], \
            format_type = "culled")
        figurepath = os.path.join(main_dir, 'graphics', 'table_culled_model_'+m+'_mqe.pdf')
        plt.savefig(figurepath, dpi = 400)
        plt.close()
    
    # Make figures (with metrics as column headers)
    fig, ax = make_tab_big(means, means_col, figsize = (7, 10))
    figurepath = os.path.join(main_dir, 'graphics', 'table_metrics_mean.pdf')
    plt.savefig(figurepath, dpi = 400)
    plt.close()
    
    medians['Livestock culled'] = medians['Livestock culled']/1000.
    fig, ax = make_tab_big(medians, medians_col, figsize = (7, 10))
    figurepath = os.path.join(main_dir, 'graphics', 'table_metrics_median.pdf')
    plt.savefig(figurepath, dpi = 400)
    plt.close()
    
    fig, ax = make_tab_big(quantiles, quantiles_col, figsize = (7, 10))
    figurepath = os.path.join(main_dir, 'graphics', 'table_metrics_quantile90.pdf')
    plt.savefig(figurepath, dpi = 400)
    plt.close()
    
    fig, ax = make_tab_big(variances, variances_col, figsize = (7, 10))
    figurepath = os.path.join(main_dir, 'graphics', 'table_metrics_variance.pdf')
    plt.savefig(figurepath, dpi = 400)
    plt.close()
    
    # Relative measures
    
    # Create the ranking and scores for coloring the table
    
    measures = [x for x in list(cost.columns) if x not in ['control', 'model']]
    df_long_cost = pd.melt(cost, id_vars = ['model', 'control'], \
        value_vars = measures, var_name = 'Measure')
    df_long_cost['metric'] = "cost"
    
    measures = [x for x in list(duration.columns) if x not in ['control', 'model']]
    df_long_dur = pd.melt(duration, id_vars = ['model', 'control'], \
        value_vars = measures, var_name = 'Measure')
    df_long_dur['metric'] = "duration"
    
    measures = [x for x in list(culled.columns) if x not in ['control', 'model']]
    df_long_cull = pd.melt(culled, id_vars = ['model', 'control'], \
        value_vars = measures, var_name = 'Measure')
    df_long_cull['metric'] = "livestock culled"
    
    df_long_full = pd.concat([df_long_cost, df_long_dur, df_long_cull], \
        ignore_index = True)
    
    rel = lambda x: (x - np.min(x) )/np.ptp(x)
    
    relative = df_long_full.groupby(['metric', 'model', 'Measure'])['value'].apply(rel)
    df_long_full['rel'] = relative
    
    df_long_full.pivot_table(index = ['control', 'model', 'metric'], 
        columns = 'Measure')['rel']
    
    # Marker map
    marker_map = dict({"ip": ".", \
        "dc":"*", \
        "rc":"o", \
        "v03": "x", \
        "v10": "+", \
        "v03_2kill": "s",\
        "v10_2kill": "D"})
    df_long_full['marker'] = df_long_full.control.map(marker_map)
    
    # Create the scatterplots (using a common metric)
    for metric in ['cost', 'duration', 'livestock culled']:
        for mod in ['A', 'B', 'C', 'D', 'E']:
            
            fig, ax = scatterplot(df_long_full, group_col = 'metric', \
                group = metric, variable_col = 'Measure', model = mod, figsize = (8,8))
            
            figname = 'scatter_metric_'+metric+'_model_'+mod+'.pdf'
            figurepath = os.path.join(main_dir, 'graphics', figname)
            plt.savefig(figurepath, dpi = 400)
            plt.close()
    
    # Create scatterplots (using a common statistic)
    df_long_f = df_long_full.loc[df_long_full.control != "v10_2kill" ,]
    for meas in ['Mean', 'Median', 'Quantile90', 'Variance']:
        for mod in ['A', 'B', 'C', 'D', 'E']:
            
            fig, ax = scatterplot(df_long_f, group_col = 'Measure', \
                group = meas, variable_col = 'metric', model = mod)
            
            figname = 'scatter_stat_'+meas+'_model_'+mod+'.pdf'
            figurepath = os.path.join(main_dir, 'graphics', figname)
            plt.savefig(figurepath, dpi = 400)
            plt.close()
    
    # Create single scatterplots for outbreak duration versus livestock culled
    for plot_type in ['value', 'rel']:
        fig, ax = plt.subplots(2, 3, figsize = (18, 12))
        for i, mod in zip([0, 1, 3, 4, 5], ['A', 'B', 'C', 'D', 'E']):
            row = i/3
            col = np.mod(i, 3)
        
            # Turn on x axis for bottom row subplots only
            if row == 1:
                x_lab_on = True
            else:
                x_lab_on = False
        
            # Turn on y axis for LHS subplots
            if col == 0:
                y_lab_on = True
            else:
                y_lab_on = False
                
            if plot_type == 'value':
                ax_fontsize = 26
                ledge_fontsize = 20
                ticksize = 18
                fig_x = 35
                fig_y = 20
            else:
                ax_fontsize = 34
                ledge_fontsize = 24
                ticksize = 26
                fig_x = 18
                fig_y = 12
                
            single_scatterplot(df_long_full, 'Measure', 'Mean', \
                'metric', 'duration', 'livestock culled', mod, \
                ax[row, col], x_lab_on, y_lab_on, ticksize = ticksize, \
                var_to_plot = plot_type)
        
        plt.figtext(.5,.04,'Outbreak duration',\
            fontsize = ax_fontsize, ha='center', va = 'center')
        
        plt.figtext(.04,.5,'Livestock culled',\
            fontsize = ax_fontsize, va = 'center', ha='center', \
            rotation = 'vertical')
        
        fig.set_size_inches(fig_x, fig_y)
        
        create_legend(ax[0,2], fontsize = ledge_fontsize, markersize = 20)
        
        figname = 'scatter_single_model_all_'+plot_type+'.pdf'
        figurepath = os.path.join(main_dir, 'graphics', figname)
        plt.savefig(figurepath, dpi = 400)
        plt.close()
    
    # Create the custom legend
    fig = plt.figure(figsize = (6, 6), frameon=False)
    ax = fig.add_subplot(111)
    create_legend(ax, markersize = 12)
    plt.savefig(os.path.join(main_dir, "graphics","legend_all.pdf"), dpi = 400)
    plt.close()
    
    ############################################
    # Scatterplot of the landscape
    ############################################
    
    df = pd.read_csv(os.path.join(main_dir, 
        "data", "RAPIDD_synthetic_population.csv"))
    df['tot'] = df['cattle'] + df['pigs'] + df['sheep']
    df = df[df['tot'] != 0]
    df['id1'] = np.arange(len(df))+1
    df = df.set_index('id1')
    
    df['color'] = 'green'
    
    fig = plt.figure(figsize = (10,10), frameon = False)
    ax = fig.add_subplot(111)
    ax.set_aspect('equal')
    scatter_plot = ax.scatter(df.easting, df.northing, s = 3)
    scatter_plot.set_facecolor(df.color)
    scatter_plot.set_edgecolor('face')
    ax.set_xlabel('Easting', fontsize = 20)
    ax.set_ylabel('Northing', fontsize = 20)
    plt.axis('off')
    plt.savefig(os.path.join(main_dir, 'graphics', 'landscape.pdf'), \
        dpi = 600, bbox_inches = 'tight', pad_inches = 0)

#### Output the legend
def create_legend(ax, fontsize = 14, markersize = 15):
    
    plt.rc('legend',**{'fontsize':fontsize})
    
    models = mpatches.Patch(color=(1,1,1,1), label='Model')
    
    Apatch = mpatches.Patch(color='#984EA3', label='A', alpha = 0.45, \
        edgecolor = 'none')
    Bpatch = mpatches.Patch(color='#FF7F00', label='B', alpha = 0.45, \
        edgecolor = 'none')
    Cpatch = mpatches.Patch(color='#4DAF4A', label='C', alpha = 0.45, \
        edgecolor = 'none')
    Dpatch = mpatches.Patch(color='#377EB8', label='D', alpha = 0.45, \
        edgecolor = 'none')
    Epatch = mpatches.Patch(color='#E41A1C', label='E', alpha = 0.45, \
        edgecolor = 'none')
    
    line_props = {'markersize': markersize, 'linestyle': 'None', 'alpha': 0.7,\
        'color':'black'}
    
    controls = mpatches.Patch(color=(1,1,1,1), label='Control')
    v10l = mlines.Line2D([],[], marker='+',label='V10L', **line_props)
    v10k = mlines.Line2D([], [], marker='D', label='V10K', **line_props)
    v3l = mlines.Line2D([], [], marker='x', label='V3L', **line_props)
    v3k = mlines.Line2D([], [], marker='s',label='V3K', **line_props)
    rc = mlines.Line2D([], [], marker='o', label='RC', **line_props)
    ip = mlines.Line2D([], [], marker='.',label='IP', **line_props)
    dc = mlines.Line2D([], [], marker='*', label='DC', **line_props)
    
    space = mpatches.Patch(color=(1,1,1,1), label='')
    
    all_handles = [models, Apatch,  Bpatch, Cpatch, Dpatch, Epatch, \
        space, space, controls, ip, dc, rc, v3l, v3k, v10l, v10k]
    
    legend_all = ax.legend(handles = all_handles, \
        loc = 10, numpoints = 1, frameon = False, ncol = 2, \
        framealpha = 0.0)
    
    ax.set_frame_on(False)
    ax.set_xticks([])
    ax.set_xticklabels([])
    ax.set_yticks([])
    ax.set_yticklabels([])
    ax.add_artist(legend_all)
    return ax


def create_table(df, obj, thresh):
    
    emp = lambda x: float(np.sum(x > thresh))/len(x)
    
    dims = ['control', 'model']
    
    tab = pd.concat([ \
    df.groupby(dims)[obj].mean(), \
    df.groupby(dims)[obj].median(), \
    df.groupby(dims)[obj].var(), \
    df.groupby(dims)[obj].quantile(q = 0.9), \
    df.groupby(dims)[obj].apply(emp)], axis = 1)
    
    col_names = ["Mean", "Median", "Variance", \
        "Quantile90", "EmpProb"+str(int(thresh))]
    
    tab.columns = col_names
    return tab


def manipulate_table(df):
    """
    Create ranks and color scheme table.  
    
    """
    measures = list(df.columns[2:7])
    
    # Add the missing Model B control measure (rc)
    column_names = ["control"]
    column_names.extend(list(df.columns[1:7]))
    
    rc = dict(zip(column_names, \
        ["rc", "B", np.nan, np.nan, np.nan, np.nan, np.nan]))
    df = pd.concat([df, \
        pd.DataFrame(rc, index = [df.shape[0]])], ignore_index = True)

    # Change "ip" to "aip" so alphabetical ordering wins out
    df = df.replace("ip", "aip")
    df = df.sort(columns = ['control', 'model'])
    
    ############
    # Create the ranking and scores for coloring the table
    
    df_long = pd.melt(df, id_vars = ['model', 'control'], \
        value_vars = measures, var_name = 'Measure')
    
    df_long['Rank'] = df_long.groupby(['model', \
        'Measure'])['value'].rank(ascending = True, 
        method = 'min', na_option = 'keep')
    
    max_group = df_long.groupby(['model', 'Measure'])['Rank'].max()
    df_long = df_long.join(max_group, on=['model', 'Measure'], rsuffix='_r')
    
    df_long['c_index'] = (df_long.Rank - 1)/(df_long.Rank_r - 1)
    
    df_long.control = pd.Categorical(df_long.control, \
        categories = ["aip", "dc", "rc", \
        "v03", "v03_2kill", "v10", "v10_2kill"])
    
    df_long.Measure = pd.Categorical(df_long.Measure, \
        categories = measures)
    
    df_long = df_long.sort(columns = ['Measure', 'control', 'model'])
    
    
    value_tab = df_long.pivot_table(values = 'value', \
        index = ['control', 'model'], columns = ['Measure'])
    
    value_tab = np.around(value_tab[measures], decimals = 2)
    
    col_tab = df_long.pivot_table(values = 'c_index', \
        index = ['control', 'model'], columns = ['Measure'])
    col_tab = col_tab[measures]
    
    return value_tab, col_tab


def make_tab_dur(value_tab, col_tab, cmap = ["#4575b4", "#ffffbf", "#d73027"]):
    """
    Make table with values and colors
    
    """
    
    measures = list(value_tab.columns)
    
    # User-defined colormap from blue to red, going through a darker colour
    cm_new = mcol.LinearSegmentedColormap.from_list("custom", cmap)
    cnorm = mcol.Normalize(vmin = 0, vmax = 1)
    cpick = cm.ScalarMappable(norm = cnorm, cmap = cm_new)
    
    fig, ax = plt.subplots()
    ax.set_axis_off()
    
    # Create the table (so that you can avoid colouring the ranks manually)
    tb = Table(ax, bbox=[0,0,1,1])
    
    nrows, ncols = value_tab.shape
    width, height = 1.0 / (ncols+1), 1.0 / (nrows+1)
    
    for (i, j), val in np.ndenumerate(value_tab):
        colour = cpick.to_rgba(col_tab.ix[i, j])
        alpha = 0.7
        colour = (colour[0], colour[1], colour[2], alpha)
        text_colour = "White"#cpick.to_rgba(1 - col_tab.ix[i, j])
        
        # Find the best text colour.  
        # If green is greater than 190/255 then use black.  
        # unless blue is greater than 150/255.  
        
        #[a < 0.9 for (r, g, b, a) in colour]
        if (colour[1] > 190./255) & (colour[2] > 150./255):
            text_colour = "Black"
        
        if np.isnan(val):
            colour = "White"
            text_colour = "Black"
        
        # Format value string
        formatted_val = formatval(val, measures[j], "duration")
        
        # Add cell to the table
        tb.add_cell(i, j + 1, width, height, text = formatted_val, 
                    loc = 'right', facecolor = colour, edgecolor = 'none')
        
        # Set the font colour
        tb._cells[(i, j + 1)]._text.set_color(text_colour)
        
    # Columns
    for j, label in enumerate(measures):
        tb.add_cell(-1, j + 1, width, height, text = label, loc = 'center', 
                           edgecolor = 'none', facecolor = 'none')
        tb._cells[(-1, j+1)]._text.set_weight('bold')
    
    # 'Control' and 'Model' labels
    tb.add_cell(-1, 0, width, height, text = 'Model', loc = 'center', 
                       edgecolor = 'none', facecolor = 'none')
    tb._cells[(-1, 0)]._text.set_weight('bold')
    
    # 'Control' and 'Model' labels
    tb.add_cell(-1, -1, width, height, text = '            Control', \
        loc = 'center', edgecolor = 'none', facecolor = 'none')
    tb._cells[(-1, -1)]._text.set_weight('bold')
    
    # Rows
    for i, label in enumerate(value_tab.index.labels[1]):
        tb.add_cell(i, 0, width, height, \
        text = value_tab.index.levels[1][label], \
        loc='center', edgecolor='none', facecolor='none')
    
    # Add table to the current axes
    ax.add_table(tb)
    
    # Bigger rows
    controls = [
    "Infected\npremises\nculling\n(IP)",
    "Dangerous\ncontacts\nculling\n(DC)",
    "Ring culling\nat 3km\n(RC)",
    "Vaccinate\nat 3km\n(V3L/V3K)",
    "Vaccinate\nat 10km\n(V10L/V10K)"]
    
    incr = float(len(controls))
    
    for k, label in enumerate(controls):
        ax.text(-0.05, 1 - (k+1)/incr + 1./15., label, fontsize = 10, \
             horizontalalignment = 'center', verticalalignment = 'center')
        
        ax.axhline(k/incr, -0.15, 1,c = "Black", clip_on = False)
    
    ax.axhline(1., -0.15, 1,c = "Black", clip_on = False)
    
    #ax.text(-0.05, 1 + 1./70 , 'Control', fontsize = 12, fontweight = 'bold',\
    #     horizontalalignment = 'center', verticalalignment = 'center')
    
    fig.set_size_inches(10, 8)
    
    return fig, ax


def make_tab_big(value_tab, col_tab, format_type = "cost", \
        cmap = ["#4575b4", "#ffffbf", "#d73027"], figsize = (10,10)):
    """
    
    
    """
    
    measures = list(value_tab.columns)
    
    # User-defined colormap from blue to red, going through a darker colour
    cm2 = mcol.LinearSegmentedColormap.from_list("new", \
        cmap)
    cnorm = mcol.Normalize(vmin = 0, vmax = 1)
    cpick = cm.ScalarMappable(norm = cnorm, cmap = cm2)
    
    fig, ax = plt.subplots()
    ax.set_axis_off()
    
    # Create the table (so that you can avoid colouring the ranks manually)
    tb = Table(ax, bbox=[0,0,1,1])
    
    nrows, ncols = value_tab.shape
    width, height = 1.0 / (ncols+1), 1.0 / (nrows+1)
    
    for (i, j), val in np.ndenumerate(value_tab):
        colour = cpick.to_rgba(col_tab.ix[i, j])
        alpha = 0.7
        colour = (colour[0], colour[1], colour[2], alpha)
        text_colour = "White" #cpick.to_rgba(1 - col_tab.ix[i, j])
        
        # Find the best text colour.  
        # If green is greater than 190/255 then use black.  
        # unless blue is greater than 150/255.  
        
        #[a < 0.9 for (r, g, b, a) in colour]
        if (colour[1] > 190./255) & (colour[2] > 150./255):
            text_colour = "Black"
        
        if np.isnan(val):
            colour = "White"
            text_colour = "Black"
        
        #if measures[j] == "Livestock culled":
        #    format_type = "culled"
        #else:
        #    format_type = "cost"
        
        # Format value string
        formatted_val = formatval(val, measures[j], format_type)
        
        # Add cell to the table
        tb.add_cell(i, j+1, width, height, text = formatted_val, 
                    loc = 'right', facecolor = colour, edgecolor = 'none')
        
        # Set the font colour
        tb._cells[(i, j+1)]._text.set_color(text_colour)
        
    # Columns
    for j, label in enumerate(measures):
        tb.add_cell(-1, j+1, width, height, text=label, loc='center', 
                           edgecolor='none', facecolor='none')
        tb._cells[(-1, j+1)]._text.set_weight('bold')
    
    # 'Control' and 'Model' labels
    tb.add_cell(-1, 0, width, height, text='Model', loc='center', 
                       edgecolor='none', facecolor='none')
    tb._cells[(-1, 0)]._text.set_weight('bold')
    
    tb.add_cell(-1, -1, width, height, text='            Control', loc='center', 
                       edgecolor='none', facecolor='none')
    tb._cells[(-1, -1)]._text.set_weight('bold')
    
    # Rows
    for i, label in enumerate(value_tab.index.labels[1]):
        tb.add_cell(i, 0, width, height, \
        text = value_tab.index.levels[1][label], \
        loc='center', edgecolor='none', facecolor='none')
    
    # Add table to the current axes
    ax.add_table(tb)
    
    # Bigger rows
    controls = [
    "Infected\npremises\nculling\n(IP)",
    "Dangerous\ncontacts\nculling\n(DC)",
    "Ring culling\nat 3km\n(RC)",
    "",
    "",
    "",
    ""]
    
    incr = float(len(controls))
    
    for k, label in enumerate(controls):
        ax.text(-0.05, 1 - (k+1)/incr + 1./15., label, fontsize = 10, \
             horizontalalignment = 'center', verticalalignment = 'center')
        if k < 4:
            pass
        else:
            ax.axhline(k/incr, -0.15, 1,c = "Black", clip_on = False)
    
    ax.axhline(2./incr, -0.15, 1,c = "Black", clip_on = False)
    ax.axhline(0./incr, -0.15, 1,c = "Black", clip_on = False)
    ax.axhline(1, -0.15, 1, c = "Black", clip_on = False)
    
    ax.axhline(3./incr, -0.02, 1,c = "Black", clip_on = False)
    ax.axhline(1./incr, -0.02, 1,c = "Black", clip_on = False)
    
    ax.text(-0.1, 1 - 4./incr, "Vaccinate\n at 3km", \
        fontsize = 10, horizontalalignment = 'center', \
        verticalalignment = 'center')
    
    ax.text(0.01, 1 - 4./incr + 1./15., "to live\n(V3L)", \
        fontsize = 10, horizontalalignment = 'center', \
        verticalalignment = 'center')
    
    ax.text(0.01, 1 - 5./incr + 1./15., "to kill\n(V3K)", \
        fontsize = 10, horizontalalignment = 'center', \
        verticalalignment = 'center')
    
    ax.text(-0.1, 1 - 6./incr, "Vaccinate\n at 10km", \
        fontsize = 10, horizontalalignment = 'center', \
        verticalalignment = 'center')
    
    ax.text(0.01, 1 - 6./incr + 1./15., "to live\n(V10L)", \
        fontsize = 10, horizontalalignment = 'center', \
        verticalalignment = 'center')
    
    ax.text(0.01, 1 - 7./incr + 1./15., "to kill\n(V10K)", \
        fontsize = 10, horizontalalignment = 'center', \
        verticalalignment = 'center')
    
    # Add the extra lines
    #ax.axhline(k/incr, -0.15, 1,c = "Black", clip_on = False)
    
    #ax.text(-0.05, 1 + 1./70 , 'Control', fontsize = 12, fontweight = 'bold',\
    #     horizontalalignment = 'center', verticalalignment = 'center')
    
    fig.set_size_inches(figsize)
    return fig, ax


def formatval(x, column_type, format_type):
    """
    Format numbers nicely for a table
    x is a float
    """
    if np.isnan(x):
        # Format nan values
        new_string = "NA"
    else:
        if float(x) <= 1.0:
            if column_type[0:7] == 'EmpProb':
                # For the final column of empirical probabilities, use 2dp
                new_string = '{:.2f}'.format(x)
            else:
                # If the value is <1 and not in the emp prob column then
                # write "< 1"
                new_string = "< 1"
        else:
            if (format_type == "culled"):
                # Scientific notation for the values of culled livestock
                #new_string = '{:.2e}'.format(x)
                # Express in thousands
                new_string = '{:.0f}'.format(x/1000.)
            else:
                # No decimal places
                new_string = '{:.0f}'.format(x)
    return(new_string)


def scatterplot(df, group_col, group, variable_col, model, figsize = (6,6)):
    """
    df : long dataframe
    
    """
    
    df = df.loc[df[group_col] == group]
    
    if model == "E":
        colr = "#E41A1C"#"#F8766D" # RED
    elif model == "C":
        colr = "#4DAF4A"#"#00BA38" # GREEN
    elif model == "D":
        colr = "#377EB8"#"#619CFF" # BLUE
    elif model == "A":
        colr = "#984EA3"#"#9750C8" # PURPLE
    elif model == "B":
        colr = "#FF7F00" #808080 # ORANGE
    
    measures = list(df[variable_col].unique())
    n_measures = len(measures)
    
    fig, ax = plt.subplots(n_measures, n_measures, figsize = figsize)
    r = 0
    for row in measures:
        c = 0
        for col in measures:
            if row == col:
                ax[r, c].axis('off')
                ax[r, c].grid(b = 'off')
                title = str(np.char.capitalize(row)).replace(" ", " \n")
                ax[r, c].text(0.5, 0.3, title, fontsize = 11, 
                    horizontalalignment = 'center', 
                    verticalalignment = 'center', 
                    fontweight = 'bold')
            elif c > r:
                ax[r, c].axis('off')
                ax[r, c].grid(b = 'off')
            else:
                #print row
                #print col
                x = df.loc[(df.model == model) & (df[variable_col] == col), 'rel']
                y = df.loc[(df.model == model) & (df[variable_col] == row),  'rel']
                markers = df.loc[(df.model == model) & (df[variable_col] == row),  'marker']
                for p, q, m in zip(x, y, markers):
                    ax[r, c].scatter(p, q, 
                        marker = m, \
                        s = 120, \
                        c = colr, \
                        edgecolor = 'black', \
                        linewidth = (1,),
                        alpha = 0.6)
                
                ax[r, c].xaxis.set_tick_params(size = 1)
                ax[r, c].yaxis.set_tick_params(size = 1)
            
                ax[r, c].xaxis.set_ticklabels([0, 0.5, 1])
                ax[r, c].yaxis.set_ticklabels([0, 0.5, 1])
            
                ax[r, c].set_xlim([-0.1, 1.1])
                ax[r, c].set_ylim([-0.1, 1.1])
            
                ax[r, c].grid(b = 'off')
                ax[r, c].set_aspect('equal')
                ax[r, c].set_axis_bgcolor('white')
            
                for tick in ax[r,c].xaxis.get_major_ticks():
                    tick.label.set_fontsize(8)
            
                for tick in ax[r,c].yaxis.get_major_ticks():
                    tick.label.set_fontsize(8)
            
                if (c == 0) | (r == (len(measures)-1)) & (col == 0):
                    # If these are border squares, then put ticks on them.
                    ax[r, c].yaxis.set_ticks([0, 0.5, 1])
                    ax[r, c].yaxis.set_ticklabels([0, 0.5, 1])
                else:
                    ax[r, c].yaxis.set_tick_params(size=0)
                    ax[r, c].yaxis.set_ticks([])
            
                if (r == (len(measures)-1)) | (r == (len(measures)-1)) & (c == 0):
                    ax[r, c].set_xticklabels([0, 0.5, 1])
                    ax[r, c].set_xticks([0, 0.5, 1])
                else:
                    ax[r, c].xaxis.set_tick_params(size=0)
                    ax[r, c].xaxis.set_ticks([])
                    ax[r, c].autoscale_view('tight')
            c += 1
        r += 1
    fig.suptitle('Model '+model, fontsize = 20, fontweight = 'bold')
    plt.subplots_adjust(wspace = 0.05, hspace = 0.1, left = 0.1, right = 1.0, bottom = 0.1, top = 0.95)
    return fig, ax


def single_scatterplot(df, group_col, group, variable_col, var1, var2, model, ax, xlab, ylab, ticksize = 18, var_to_plot = 'rel'):
    #df, col_names, measure1, measure2, model):
    """
    Produce a single scatterplot of two relative measures.  
    
    Parameters
    ----------
    
    col_names:  designating the control strategies to plot
    measure1:   First measure to plot (x axis)
    measure2:   Second measure to plot (y axis)
    mode:       which model to use (for colouring purposes)
    
    """
    
    df = df.loc[df[group_col] == group]
    
    if model == "E":
        colr = "#E41A1C"#"#F8766D" # RED
    elif model == "C":
        colr = "#4DAF4A"#"#00BA38" # GREEN
    elif model == "D":
        colr = "#377EB8"#"#619CFF" # BLUE
    elif model == "A":
        colr = "#984EA3"#"#9750C8" # PURPLE
    elif model == "B":
        colr = "#FF7F00" #808080 # ORANGE
    
    x = df.loc[(df.model == model) & (df[variable_col] == var1), var_to_plot]
    y = df.loc[(df.model == model) & (df[variable_col] == var2),  var_to_plot]
    markers = df.loc[(df.model == model) & (df[variable_col] == var1),  'marker']
    
    for p, q, m in zip(x, y, markers):
        ax.scatter(p, q, 
            marker = m, \
            s = 400, \
            c = colr, \
            edgecolor = 'black', \
            linewidth = (1,),
            alpha = 0.6)
    
    ax.xaxis.set_tick_params(size = 5)
    ax.yaxis.set_tick_params(size = 5)
    
    if var_to_plot == 'rel':
        ax.xaxis.set_ticklabels([0, 0.5, 1])
        ax.yaxis.set_ticklabels([0, 0.5, 1])
    
        ax.yaxis.set_ticks([0, 0.5, 1])
        ax.xaxis.set_ticks([0, 0.5, 1])
    
        ax.set_xlim([-0.1, 1.1])
        ax.set_ylim([-0.1, 1.1])
        
        ax.set_aspect('equal')
        
        ax.xaxis.set_visible(xlab)
        ax.yaxis.set_visible(ylab)
        
        plt.subplots_adjust(wspace = 0.05, \
            hspace = 0.1, \
            left = 0.1, \
            right = 1.0, \
            bottom = 0.1, \
            top = 0.95)
    else:
        import matplotlib.ticker as mtick
        ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))
        
        plt.subplots_adjust(wspace = 0.15, \
            hspace = 0.1, \
            left = 0.1, \
            right = 0.95, \
            bottom = 0.1, \
            top = 0.95)
    
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(ticksize)
    
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(ticksize)
    
    #ax.set_xlabel(np.char.capitalize(var1), size = 20)
    #ax.set_ylabel(np.char.capitalize(var2), size = 20)
    
    ax.grid(b = 'off')
    ax.set_axis_bgcolor('white')
    
    return ax

if __name__ == "__main__":
    run_analysis()
