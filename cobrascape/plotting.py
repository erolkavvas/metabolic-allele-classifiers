### Plotting utilities for cobrascape
import cobrascape.ensemble as ens
from random import shuffle
import numpy as np
import seaborn as sns
import pandas as pd
import warnings ### Freaking SKlearn bro gives hella warnings.
warnings.filterwarnings("ignore", category=DeprecationWarning)

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib import cm
from matplotlib.text import Annotation
from mpl_toolkits.mplot3d.proj3d import proj_transform
from scipy.interpolate import griddata
# import matplotlib.cm as cm
from matplotlib.colors import Normalize
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler


### ---------------------------------------------------
### --------- GWAS plotting functionality  ------------
### ---------------------------------------------------
def popfva_manhatten_plot(popfva_enrich_dic, s_obj, pheno_id="isoniazid", fdr_line=True,
                          s_size=100, labelsizes=20, f_scale=1.0, figSIZE=(9,4)):
    """Plots a manhatten plot for both alleles and popFVA associations
        - popfva_enrich_dict = {pheno_id: dataframe of popfva enrichments}
        - s_obj = cobrascape Sample object
        - f_scale = fontsize
    """
    AMR_to_gene = {"ethambutol": ["Rv3806c", "Rv3795"], "pyrazinamide": ["Rv2043c"],
                   "isoniazid": ["Rv1908c", "Rv1484"], 
                   ## "isoniazid": ["Rv1908c","Rv1484","Rv2243","Rv2247","Rv2245","Rv3139","Rv1483","Rv0129c"], 
                   "4-aminosalicylic_acid": ["Rv2447c", "Rv2764c"]}
    # sns.set()
    plt.style.use(["seaborn-white"])
    sns.set_style("ticks")
    flatui = ["#e74c3c", "#3498db", "#9b59b6",   "#ff7f00",  "lightgray"]
    current_palette = sns.color_palette(flatui)
    amr_to_color = dict(zip(["isoniazid", "pyrazinamide", "ethambutol", 
                             "4-aminosalicylic_acid", "unknown"], current_palette.as_hex()))
    
    rc_par = {"axes.labelsize": labelsizes, "xtick.labelsize":labelsizes, 
              "ytick.labelsize":labelsizes,"axes.titlesize":labelsizes}
    with sns.plotting_context("notebook", font_scale=f_scale, rc=rc_par):
        fig, (ax_gwas, ax_popfva) = plt.subplots(1, 2, figsize=figSIZE)
        X_alleles = s_obj.x_allele_dict[pheno_id]
        Y_pheno = s_obj.y_pheno_dict[pheno_id]
        ### ---- popFVA manhatten plot -----
        sample_inference_df = s_obj.anova_dict[pheno_id].copy()
        sample_inference_df.fillna(1,inplace=True)
        sample_inference_df["AIC"]=sample_inference_df.index.map(lambda x: s_obj.assess_df.loc[x,"AIC_"+pheno_id])
        sample_inference_df["BIC"]=sample_inference_df.index.map(lambda x: s_obj.assess_df.loc[x,"BIC_"+pheno_id])

        react_to_gene, react_to_AMR = {}, {}
        feat_to_test_df = popfva_enrich_dic[pheno_id]
        for y in feat_to_test_df.index:
            react = y.replace("_max", "").replace("_min", "")
            rxn_genes = [str(x.id) for x in s_obj.base_cobra_model.reactions.get_by_id(react).genes]
            react_to_gene.update({y: rxn_genes})
            for rg in rxn_genes:
                for dr, dr_g in AMR_to_gene.items():
                    if rg in dr_g:
                        react_to_AMR.update({y: dr})

        popfva_genes = list(set([str(x.split("_")[0]) for x in X_alleles.columns]))
        feat_to_test_df["drug_type"]= feat_to_test_df.index.map(lambda x: react_to_AMR[x] if x in react_to_AMR.keys() else "unknown")
        feat_to_test_df["log10pval"]= -np.log(feat_to_test_df["pvalue"])
        x = [i for i in range(len(feat_to_test_df.index))]
        shuffle(x)
        feat_to_test_df["rand_index"] = x
        new_color_order = [amr_to_color[x] for x in feat_to_test_df["drug_type"].unique()]
        ax_popfva = sns.scatterplot(x="rand_index", y="log10pval", data=feat_to_test_df, hue="drug_type",
                         palette=new_color_order, ax=ax_popfva, s=s_size)
        ax_popfva.set_title("popFVA enrichments: "+pheno_id)
        ax_popfva.set_xlabel("popFVA features of "+str(len(popfva_genes))+" AMR genes")
        ax_popfva.legend(loc='upper right', bbox_to_anchor=(1.8, 0.9))

        ### ---- Allele manhatten plot ---- 
        anova_df = ens.compute_ANOVA_test(X_alleles,Y_pheno)
        anova_df = ens.FDR(anova_df, fdr_rate=1) ### Add horizontal lines for FDR
        anova_genes = list(set([str(x.split("_")[0]) for x in anova_df.index]))
        allele_to_AMR = {}
        for drg, drg_genes in AMR_to_gene.items():
            for allele in anova_df.index:
                if allele.split("_")[0] in drg_genes:
                    allele_to_AMR.update({allele: drg})

        anova_df["drug_type"]=anova_df.index.map(lambda x: allele_to_AMR[x] if x in allele_to_AMR.keys() else "unknown")
        anova_df["log10pval"]= -np.log(anova_df["pvalue"])
        x = [i for i in range(len(anova_df.index))]
        shuffle(x)
        anova_df["rand_index"] = x
        new_color_order = [amr_to_color[x] for x in anova_df["drug_type"].unique()]

        ax_gwas = sns.scatterplot(x="rand_index", y="log10pval", data=anova_df, hue="drug_type",
                                  ax=ax_gwas, palette=new_color_order, s=s_size, legend=False)
        ax_gwas.set_title("classical GWAS: "+pheno_id)
        ax_gwas.set_xlabel("alleles of "+str(len(popfva_genes))+" AMR genes")
    return fig


### ---------------------------------------------------
### --- Plotting gene-gene constraint interactions  ---
### ---------------------------------------------------
def plot_ac_interactions(COBRA_MODEL, gene_rxn_action_dict, x_gene_rxn, y_gene_rxn):
    """Generates a 2 by 2 heatmaps of allele constraint interactions for min/max optimizations
        of corresponding allele-encoded metabolic reactions.
    input: 
        COBRA_MODEL, x_gene_rxn, y_gene_rxn
        gene_rxn_action_dict: 
    output:
        f: figure of 2 by 2 heatmaps
        payoff_df_dict: dictionary of heatmap values for each optimized objective
    -- example: f, payoff_dict = get_ac_interactions(COBRA_MODEL, ['Rv1908c','CAT'], ['Rv3280','ACCC'])
    """
    f, (ax1, ax2) = plt.subplots(2, 2, figsize=(8,6),sharex=True, sharey=True)
    interact_landscape_list = []
    interact_landscape_dict = {}
    COBRA_MODEL_COPY = COBRA_MODEL.copy()
    payoff_df_dict = {}
    for obj_id, ax in [(x_gene_rxn[1], ax1), (y_gene_rxn[1], ax2)]:
        for obj_dir, ax_col in [("max", ax[0]),("min", ax[1])]:
            for x_action, x_constraint in gene_rxn_action_dict[x_gene_rxn[0]][x_gene_rxn[1]].items():
                for y_action, y_constraint in gene_rxn_action_dict[y_gene_rxn[0]][y_gene_rxn[1]].items():
                    with COBRA_MODEL_COPY:   
                        COBRA_MODEL_COPY.objective = obj_id
                        COBRA_MODEL_COPY.objective_direction = obj_dir
                        strain_react_x = COBRA_MODEL_COPY.reactions.get_by_id(x_gene_rxn[1])
                        if x_action.split("_")[0] == "lb":    
                            COBRA_MODEL_COPY.reactions.get_by_id(x_gene_rxn[1]).lower_bound = x_constraint     
                        elif x_action.split("_")[0] == "ub":
                            COBRA_MODEL_COPY.reactions.get_by_id(x_gene_rxn[1]).upper_bound = x_constraint

                        strain_react_y = COBRA_MODEL_COPY.reactions.get_by_id(y_gene_rxn[1])
                        if y_action.split("_")[0] == "lb":
                            COBRA_MODEL_COPY.reactions.get_by_id(y_gene_rxn[1]).lower_bound = y_constraint 
                        elif y_action.split("_")[0] == "ub":
                            COBRA_MODEL_COPY.reactions.get_by_id(y_gene_rxn[1]).upper_bound = y_constraint

                        OPT_VAL = COBRA_MODEL_COPY.optimize().objective_value
                        interact_landscape_list.append((x_action,y_action,OPT_VAL))
                        if x_action not in interact_landscape_dict.keys():
                            interact_landscape_dict[x_action] = {}
                            interact_landscape_dict[x_action].update({y_action: OPT_VAL})
                        else:
                            interact_landscape_dict[x_action].update({y_action: OPT_VAL})

            # print x_action, x_constraint
            payoff_df = pd.DataFrame(interact_landscape_dict).T
            g = sns.heatmap(payoff_df, ax=ax_col)
            g.set_xlabel(": ".join(y_gene_rxn))
            g.set_ylabel(": ".join(x_gene_rxn))
            g.set_title(obj_id+" "+obj_dir)
            payoff_df_dict.update({obj_id+" "+obj_dir: payoff_df})
    return f, payoff_df_dict

### ---------------------------------------------------
### --------- Plotting allele-constraint map  ------------
### ---------------------------------------------------
def plot_allele_constraint_map(s_obj, plot_alleles, ic_id="BIC", pheno_id="isoniazid", thresh_val=12, model_list=[],
                               FIGSIZE="auto" ,save_plot=False, gene_name=False, clusterCOL=True, clusterROW=True,
                               method="average", metric='correlation', allele_num=2, cmap_ac=sns.color_palette("PuOr", 17), 
                               vmax=6, vmin=-6, verbose=False):
                              
    """ Plot a seaborn clustermap of (hq models, alleles) where the values are the constraints
    """
    drug_allele_df = ens.filter_0_alleles(s_obj.x_allele_dict[pheno_id].copy(),allele_num=allele_num)
    if len(model_list)==0:
        top_models = s_obj.get_hq_samples(ic_id=ic_id,pheno_id=pheno_id,thresh_val=thresh_val)
        top_models_list = top_models.index.tolist()
        cs_df = s_obj.constraint_df.loc[top_models_list].copy()
    else:
        top_models_list = model_list
        cs_df = s_obj.constraint_df.loc[top_models_list].copy()

    if len(top_models_list)<2:
    	print("There are less than 2 hq models! Can't cluster... returning None")
    	return None

    if FIGSIZE=="auto":
        FIGSIZE = (len(top_models_list)/4.0, len(plot_alleles)/4.0)

    if verbose==True:  
    	print("len(top_models): ",len(top_models_list))
    	print("len(plot_alleles): ",len(plot_alleles))
    	print("FIGSIZE: ",FIGSIZE)
    
    if gene_name==True:
        drug_allele_df = s_obj.convert_gene2name_df(drug_allele_df.T)
        drug_allele_df = drug_allele_df.T
        cs_df = s_obj.convert_gene2name_df(cs_df.T)
        cs_df = cs_df.T
        
    hq_ac_df = cs_df.loc[top_models_list, plot_alleles].copy()
    # hq_ac_df = pd.concat([pd.DataFrame(top_models), cs_df.loc[top_models_list, plot_alleles]], axis=1)
    action_constraint_mapping = ens.get_action_constraint_mapping(s_obj.action_num, add_no_change=s_obj.add_no_change)
    LOR_list = []
    for x_allele in plot_alleles:
        strains_with_allele = drug_allele_df[drug_allele_df[x_allele]==1].index.tolist()
        allele_resist_percent  = round(ens.resist_percentage(s_obj.y_pheno_dict[pheno_id], strains_with_allele), 4)
        LOR, num_R = ens.log_odds_ratio(x_allele, drug_allele_df, s_obj.y_pheno_dict[pheno_id], addval=0.5)
        LOR_list.append(LOR)

    cmap = cm.RdGy_r # cm.coolwarm
    norm = Normalize(vmin=-4, vmax=4)# Normalize(vmin=-0.5, vmax=0.5)# 
    allele_color_list = [cmap(norm(x)) for x in LOR_list]

    # sns.palplot(sns.color_palette("RdBu_r", 7))
    hq_ac_df_val = hq_ac_df.replace(action_constraint_mapping)
    hq_ac_df_val.fillna(0, inplace=True)
        
    if gene_name!=True:
        allele_importName_df = s_obj.convert_gene2name_df(hq_ac_df_val[plot_alleles].T.copy())
        allele_importName_df = allele_importName_df.T
    else:
        allele_importName_df = hq_ac_df_val[plot_alleles]

    try:
        fig_acmap = sns.clustermap(allele_importName_df.T, method=method, metric=metric,
			linewidths=0.0, row_colors=allele_color_list, 
			col_cluster=clusterCOL, row_cluster=clusterROW,
			vmax=vmax, vmin=vmin, cmap=cmap_ac, figsize=FIGSIZE);
    except Exception as error:
	    print("function raised %s" % error)
	    print("... settingm method='ward', metric='euclidean'")
	    fig_acmap = sns.clustermap(allele_importName_df.T, method= "ward", metric="euclidean",
	    	linewidths=0.0, row_colors=allele_color_list, 
	    	col_cluster=clusterCOL, row_cluster=clusterROW,
	    	vmax=vmax, vmin=vmin, cmap=cmap_ac, figsize=FIGSIZE);

    if save_plot==True:
        extra_id = "_PVALCUTOFF-"+str(pval_CUTOFF)+"_NUMMODELS-"+str(len(top_models_list))+"_NUMALLELES-"+str(len(plot_alleles))
        fig_acmap.savefig("fig6/"+AMR__drug+"_ac_clustermap"+extra_id+".svg", format="svg")
        fig_acmap.savefig("fig6/"+AMR__drug+"_ac_clustermap"+extra_id+".png", format="png", dpi=300)
        
    return fig_acmap, allele_color_list


### ---------------------------------------------------
### ---- Plot PCA Components in 3D and barplots  ------
### ---------------------------------------------------
def get_rxn_list(popfva_ls_df, cobra_model):
    rxn_obj_list = []
    for rxn_obj in popfva_ls_df.columns:
        if "_max" in rxn_obj:
            rxn_id = rxn_obj.split("_max")[0]
        elif "_min" in rxn_obj:
            rxn_id = rxn_obj.split("_min")[0]
        rxn = cobra_model.reactions.get_by_id(rxn_id)
        if rxn.reversibility == False:
            if rxn.lower_bound>=0:
                #print(rxn.reaction, rxn.id+"_max")
                rxn_obj_list.append(rxn.id+"_max")
            elif rxn.lower_bound<0:
                #print(rxn.reaction, rxn.id+"_min")
                rxn_obj_list.append(rxn.id+"_min")
        else:
            rxn_obj_list.append(rxn.id+"_min")
            rxn_obj_list.append(rxn.id+"_max")
            
    rxn_obj_list = list(set(rxn_obj_list))
    return rxn_obj_list

### ---------------------------------------------------
### ---- Plot PCA Components in 3D and barplots  ------
### ---------------------------------------------------
from sklearn.preprocessing import MinMaxScaler
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
def plot_pca_3d(s_obj, sample_id=None, pheno_id="isoniazid", hq_model_rank=0, lr_pval_thresh=1.0, 
                models_df=pd.DataFrame(),
                FILTER_RXN_DIR=True, STANDARDIZE=False, figSIZE=(7, 5), ELEV_=-120,AZIM_=110, 
                alpha=0.7, s_size=20, labelsizes=10, labelpad=4.0, f_scale=1.0, show_plot=True):
    """Performs PCA and plots the dimensions of the most significant PCA components
        - significant PCA components are determined by the saved logistic regression fits
    """
    if sample_id==None and models_df.empty==True:
        sample_id = s_obj.hq_models.index[hq_model_rank]
    elif sample_id==None and models_df.empty==False:
        sample_id = models_df.index[hq_model_rank]
        
    sample_popfva_file = s_obj.popfva_file_loc+"sample_"+sample_id.split("_")[-1]+"_FVA.json"
    popfva_ls = ens.load_landscape_sample(sample_popfva_file)

    if FILTER_RXN_DIR==True:
        popfva_rxnfilt_ls = get_rxn_list(popfva_ls, s_obj.base_cobra_model)
        popfva_ls=popfva_ls[popfva_rxnfilt_ls]

    X = popfva_ls.reindex(s_obj.x_allele_dict[pheno_id].index).copy()
    y = s_obj.y_pheno_dict[pheno_id].copy()

    if STANDARDIZE==True:
        landscape_scaler = StandardScaler() # Standardization Z-score
    else:
        landscape_scaler = MinMaxScaler() # Normalization 0-1 scaling

    X_scaled = landscape_scaler.fit_transform(X)
    X = pd.DataFrame(X_scaled, index=X.index, columns=X.columns)
    
    plt.style.use(["seaborn-white"])
    sns.set_style("ticks")
    rc_par = {"axes.labelsize": labelsizes, "xtick.labelsize":labelsizes, 
                "ytick.labelsize":labelsizes,"axes.titlesize":labelsizes, "axes.labelpad": labelpad}
    with sns.plotting_context("notebook", font_scale=f_scale, rc=rc_par):
        fig = plt.figure(1, figsize=figSIZE)
        ax = Axes3D(fig, elev=ELEV_, azim=AZIM_)
        
        fig_2d, ax_2d = plt.subplots(1, figsize=figSIZE)
        #  = 
        if models_df.empty==True:
            s_obj.get_logreg_params(pval_thresh=lr_pval_thresh)
        else:
            s_obj.get_logreg_params(pval_thresh=lr_pval_thresh, hq_mods=models_df)
        mod_rxn_pca_df = s_obj.get_sample_pca(s_obj.hq_models.index[hq_model_rank], 
                                              pca_thresh=0.0, drop_pval=False)
        comp_1_index = int(mod_rxn_pca_df.loc["p_val"].index[0].split("x")[1])-1
        comp_2_index = int(mod_rxn_pca_df.loc["p_val"].index[1].split("x")[1])-1
        comp_3_index = int(mod_rxn_pca_df.loc["p_val"].index[2].split("x")[1])-1
        pca = PCA(n_components=0.9, svd_solver = 'full', whiten=True)
        X_reduced = pca.fit_transform(X)
        ax.scatter(X_reduced[:, comp_1_index], X_reduced[:, comp_2_index], X_reduced[:, comp_3_index], c=y.values,
                   cmap= plt.cm.bwr_r, #plt.cm.coolwarm,#plt.cm.Set1_r, #plt.cm.Set1, plt.cm.Set3, plt.cm.coolwarm, plt.cm.RdGy
                   edgecolor=None, s=s_size, alpha=alpha)
        ax_2d.scatter(X_reduced[:, comp_1_index], X_reduced[:, comp_2_index], c=y.values, cmap= plt.cm.bwr_r,
                      edgecolor=None, s=s_size, alpha=alpha)
        ax_2d.set_title("Top 2 significant PCA directions: \n%s"%(sample_id))
        ax_2d.set_xlabel("Eigenvector %s,\n P-value=%.2E"%(str(comp_1_index+1),mod_rxn_pca_df.loc["p_val"].values[0]))
        #ax_2d.set_ticklabels([])
        ax_2d.set_ylabel("Eigenvector %s,\n P-value=%.2E"%(str(comp_2_index+1),mod_rxn_pca_df.loc["p_val"].values[1]))
        #ax_2d.set_ticklabels([])
        
        ax.set_title("Top 3 significant PCA directions: %s"%(sample_id))
        ax.set_xlabel("Eigenvector %s,\n P-value=%.2E"%(str(comp_1_index+1),mod_rxn_pca_df.loc["p_val"].values[0]))
        ax.w_xaxis.set_ticklabels([])
        ax.set_ylabel("Eigenvector %s,\n P-value=%.2E"%(str(comp_2_index+1),mod_rxn_pca_df.loc["p_val"].values[1]))
        ax.w_yaxis.set_ticklabels([])
        ax.set_zlabel("Eigenvector %s,\n P-value=%.2E"%(str(comp_3_index+1),mod_rxn_pca_df.loc["p_val"].values[2]))
        ax.w_zaxis.set_ticklabels([])
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0));
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0));
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0));
        # ax.legend()
        plt.close()
        fig.tight_layout()
        if show_plot==True:
            plt.show()
        X_reduced_df = pd.DataFrame(X_reduced, index=X.index, columns=range(X_reduced.shape[1]))
        return fig, ax, fig_2d, ax_2d, mod_rxn_pca_df, pca, X_reduced_df #_df

    
def pca_load_barplot(pca_, rxn_pca_df_, num_pca_comp=4, num_pca_feat=20, show_plot=True):
    # if num_pca_comp<=len(rxn_pca_df_.index)
    f, ax_pca_barplot = plt.subplots(1,num_pca_comp, figsize=(num_pca_comp*3,num_pca_feat/4))
    for i, ax in zip(range(num_pca_comp), ax_pca_barplot):
        pca_nocoef_pval = rxn_pca_df_[rxn_pca_df_.loc["p_val"].index[i]].sort_values().drop(["coef", "p_val"])
        pca_top_comps = abs(pca_nocoef_pval[abs(pca_nocoef_pval)>0.10]).sort_values(ascending=False).index[:num_pca_feat]
        pca_nocoef_pval[pca_top_comps].sort_values().plot(kind="barh", ax=ax)
        ax.set_title("Eigenvector %s,\n P-value=%.2E,\n EVR:%.3f"%(str(int(rxn_pca_df_.loc["p_val"].index[i].split("x")[1])),
                                                                  rxn_pca_df_.loc["p_val"].values[i], 
                                                                  pca_.explained_variance_ratio_[int(rxn_pca_df_.loc["p_val"].index[i].split("x")[1])-1]))
    f.tight_layout()
    if show_plot==True:
        plt.show()
    # plt.close()
    return f

def pca_weightedload_barplot(rxn_pca_df_, num_pos=15, num_neg=15, show_plot=True):
    f, ax_coefpca = plt.subplots(1,1, figsize=(5,(num_pos+num_neg)/5+1))
    feat_pca_weight = rxn_pca_df_[rxn_pca_df_.columns[:]].copy()
    feat_pca_weight["sum"] = feat_pca_weight.apply(lambda x: np.dot(x, feat_pca_weight.loc["coef"]), axis=1)
    top_pos=feat_pca_weight["sum"].drop("coef").dropna().sort_values(ascending=True)[-num_pos:].index.tolist()[::-1]
    top_neg=feat_pca_weight["sum"].drop("coef").dropna().sort_values(ascending=True)[:num_neg].index.tolist()[::-1]
    # display(feat_pca_weight["sum"][top_neg+top_pos].sort_values(ascending=False))
    df_coefpca = feat_pca_weight["sum"][top_neg+top_pos].sort_values(ascending=True)
    df_coefpca.plot(kind="barh",ax=ax_coefpca)
    f.tight_layout()
    if show_plot==True:
        plt.show()
    return f, df_coefpca

### ---------------------------------------------------
### ---- Fitness landscape plotting functionality -----
### ---------------------------------------------------
# ---- 3D plot annotation functions -----
class Annotation3D(Annotation):
    '''Annotate the point xyz with text s'''

    def __init__(self, s, xyz, *args, **kwargs):
        Annotation.__init__(self,s, xy=(0,0), *args, **kwargs)
        self._verts3d = xyz        

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.xy=(xs,ys)
        Annotation.draw(self, renderer)
        
def annotate3D(ax, s, *args, **kwargs):
    '''add anotation text s to to Axes3d ax'''
    tag = Annotation3D(s, *args, **kwargs)
    ax.add_artist(tag)
    

# ---- Phenotypic phase plane plots -----
def PhPP_plot(cobra_model, rxn_1, rxn_2, prod_env_df, obj_direct="maximum", lineW=0.1, ELEVAT=45, AZIMU=-130, surf_trans=0.2, 
	          plt_type="surf", cmap_=plt.cm.Spectral_r, labelsizes=10, labelpad=4.0, f_scale=1.0, figSIZE=(10,10)):
	""" Plots phenotypic phase plane for two reactions rxn_1, rxn_2.
	    Code taken from game4all scripts file "02_plot_game4all"
	    - labelpad doesn't work. It's supposed to add space between axes labels and ticklabels.
	"""
	X = prod_env_df[rxn_1].values
	Y = prod_env_df[rxn_2].values
	phenotypic_var="flux_"+obj_direct
	Z = prod_env_df[phenotypic_var].values

	plt.style.use(["seaborn-white"])
	sns.set_style("ticks")
	rc_par = {"axes.labelsize": labelsizes, "xtick.labelsize":labelsizes, 
	"ytick.labelsize":labelsizes,"axes.titlesize":labelsizes, "axes.labelpad": labelpad}
	with sns.plotting_context("notebook", font_scale=f_scale, rc=rc_par):
		fig = plt.figure(figsize=figSIZE)
		ax = fig.gca(projection='3d')

		if plt_type == "trisurf":
			ax.plot_trisurf(X, Y, Z, linewidth=lineW, edgecolors="k", antialiased=True, cmap=cmap_, alpha=surf_trans) # plt.cm.viridis_r
	        
		elif plt_type == "surf":
			x1_min, x1_max = cobra_model.reactions.get_by_id(rxn_1).lower_bound, cobra_model.reactions.get_by_id(rxn_1).upper_bound
			x2_min, x2_max = cobra_model.reactions.get_by_id(rxn_2).lower_bound, cobra_model.reactions.get_by_id(rxn_2).upper_bound
			x1 = np.linspace(x1_min, x1_max, len(prod_env_df[rxn_1].unique()))
			y1 = np.linspace(x2_min, x2_max, len(prod_env_df[rxn_2].unique()))

			# x1 = np.linspace(prod_env_df[rxn_1].min(), prod_env_df[rxn_1].max(), len(prod_env_df[rxn_1].unique()))
			# y1 = np.linspace(prod_env_df[rxn_2].min(), prod_env_df[rxn_2].max(), len(prod_env_df[rxn_2].unique()))
			x2, y2 = np.meshgrid(x1, y1)
			z2 = griddata((prod_env_df[rxn_1], prod_env_df[rxn_2]), prod_env_df[phenotypic_var], (x2, y2), method='cubic')
			xgrid_scale = int(len(x1) / 20)
			ygrid_scale = int(len(y1) / 20)
			ax.plot_surface(x2, y2, z2, linewidth=lineW, edgecolors="k", rstride=1,cstride=1, antialiased=True, 
			cmap=cmap_, alpha=surf_trans)

			ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0));
			ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0));
			ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0));

			ax.set_xlim3d([min(set(X)), max(set(X))])
			ax.set_ylim3d([min(set(Y)), max(set(Y))])
			ax.set_zlim3d([min(set(Z)), max(set(Z))])

			ax.set_xlabel(rxn_1)
			ax.set_ylabel(rxn_2)
			ax.set_zlabel(phenotypic_var)
			ax.view_init(elev=ELEVAT, azim=AZIMU) # 30, -135 (-130)
			plt.close()
			return fig, ax

