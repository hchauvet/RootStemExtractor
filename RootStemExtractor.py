#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Created on Mon May 18 14:14:54 2015

Petite interface graphique pour traiter les tiges ou les racines

@author: hugo chauvet

Version: 18/04/2018

Change:
18/04/2018: [Hugo] correct datetime bug. Set percentdiam to 1.4 in MethodOlivier (previous 0.9). Windows system can now use thread
22/10/2017: [Hugo] Optimisation du positionnement du GUI, utilisation de Tk.grid a la place de Tk.pack
16/10/2015: [Hugo] Ajout de divers options pour la tige (supression etc..) avec menu click droit
                   +Refactorisation de plot_image et de la gestion du global_tige_id_mapper (pour gerer les suppressions)

30/07/2015: [Hugo] Correction bugs pour les racines dans libgravimacro + Ajout visu position de la moyenne de l'ange + Ajout d'options pour le temps des photos
25/05/2015: [Hugo] Ajout des menus pour l'export des moyennes par tiges et pour toutes les tiges + figures
20/05/2015: [Hugo] Première version
"""

import platform
import matplotlib
matplotlib.use('TkAgg')

#Remove zoom key shorcuts
matplotlib.rcParams['keymap.back'] = 'c'
matplotlib.rcParams['keymap.forward'] = 'v'

#from pylab import *
import matplotlib.pylab as mpl
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib.collections import LineCollection

#from matplotlib.widgets import Slider, Button

from numpy import (array, linspace, arctan2, sqrt, sin, cos, pi, timedelta64,
                   arange, hstack, diff, rad2deg, mean, argmin, zeros_like, ma,
                    cumsum, convolve, ones, exp, deg2rad)


from pylab import find
from scipy.optimize import fmin
import pandas as pd
#pd.options.display.mpl_style = 'default'

import cPickle as pkl
# implement the default mpl key bindings
from matplotlib.backend_bases import key_press_handler
from matplotlib.widgets import RectangleSelector
from new_libgravimacro import (save_results,
                           load_results, traite_tiges2, traite_tige2,
                           extract_Angle_Length, plot_sequence,
                           convert_angle, get_photo_time, Image,
                           compute_pattern_motion, ProcessImages)

from Extrawidgets import DraggableLines

import sys, os
if sys.version_info[0] < 3:
    import Tkinter as Tk, tkFileDialog
else:
    import tkinter as Tk, tkFileDialog

import re
finddigits = re.compile(r'\d+?')
from ttk import Style, Button, Frame, Progressbar, Entry, Scale
from threading import Thread
import Queue

__version__ = '18052018'

########################## GLOBAL DATA ########################################
data_out = None
imgs_out = None
img_object = None
text_object = None
tige_plot_object = None
traitement_file = None
base_pts_out = None
files_to_process = []
base_dir_path = None
cur_image=None
cur_tige = None
tiges_x = None
tiges_y = None
tiges_tailles = None
tiges_angles = None
tiges_lines = None
tiges_measure_zone = None
toptige = None
add_tige = False
nbclick = 0
base_tiges = []
btige_plt = []
btige_text = []
tige_id_mapper = {} #Pour changer le nom des tiges
tiges_colors = None
dtphoto = []
local_photo_path = False #If true the photo path is removed
thread_process = None #To store the thread that run image processing
infos_traitement = None
old_step = 0
add_dist_draw=False
dist_measure_pts = []
pixel_distance = None
cm_distance = None
scale_cmpix = None
B_data = {}
Beta_data = {}
End_point_data = {}
End_point_plot = {}
Tiges_seuil_offset = {} #Enregistrer les offset des tiges
Crops_data = []

PAS_TRAITEMENT = 0.3

def reset_graph_data():
    global img_object, text_object, tige_plot_object, End_point_plot
    global traitement_file, btige_plt, btige_text, tiges_colors

    img_object = None
    text_object = None
    tige_plot_object = None
    traitement_file = None
    btige_plt = []
    btige_text = []
    End_point_plot = {}

###############################################################################

########################## LOAD FILES #########################################
def _open_files():
    global files_to_process, base_dir_path, data_out, imgs_out, base_tiges
    global traitement_file, tige_id_mapper, btige_plt, btige_text, dtphoto, cidkey
    global scale_cmpix, B_data, Beta_data, End_point_data, Tiges_seuil_offset
    reset_graph_data()

    #TODO: Bug mac version TK and wild-cards
    #OLD, ('Images', '*.jpg,*.JPG'), ('Projet', '*.pkl')
    #ftypes = [('all files', '.*'), ('Images', '*.jpg,*.JPG'), ('Projet', '*.pkl')]

    #todo  filetypes=ftypes
    files = tkFileDialog.askopenfilenames(parent=root, title='Choisir les images a traiter')

    if files != '' and len(files) > 0:
        #base_dir_path = os.path.abspath(files[0]).replace(os.path.basename(files[0]),'')
        base_dir_path =  os.path.dirname(files[0])+'/'
        #Test si c'est un fichier de traitement ou des images qui sont chargées
        if '.pkl' in files[0]:
             ax.clear()
             ax.axis("off")
             ax.text(0.5,0.5,u'Chargement des données',ha='center',va='center',color='red',transform=ax.transAxes)
             canvas.draw()

             traitement_file = files[0]
             data_tmp = load_results(files[0])
             data_out = data_tmp
             imgs_out = [i['imgname'] for i in data_out['tiges_info']]

             files_to_process = imgs_out
             #Chargement des points de base
             if 'pts_base' in data_tmp:
                 base_tiges = data_tmp['pts_base']
             else:
                 base_tiges = [[(data_out['tiges_data'].xc[ti,0,0],data_out['tiges_data'].yc[ti,0,0])]*2 for ti in range(len(data_out['tiges_data'].yc[:,0,0]))]

             btige_plt = [None]*len(base_tiges)
             btige_text = [None]*len(base_tiges)
             #chargement des numeros des tiges
             if os.path.isfile(base_dir_path+'tige_id_map.pkl'):
                 with open(base_dir_path+'tige_id_map.pkl', 'rb') as f:
                     datapkl = pkl.load(f)
                     tige_id_mapper = datapkl['tige_id_mapper']

                     try:
                         scale_cmpix = datapkl['scale_cmpix']
                     except:
                         pass

                     try:
                         B_data = datapkl['B_data']
                     except:
                         pass

                     try:
                         Beta_data = datapkl['Beta_data']
                     except:
                         pass

                     try:
                         End_point_data = datapkl['End_point_data']
                     except:
                         pass

                     try:
                        Tiges_seuil_offset = datapkl['Tiges_seuil_offset']
                     except:
                        pass

             #Creation du tige_id_mapper
             for it in range(len(base_tiges)):
                 if not it in tige_id_mapper:
                     tige_id_mapper[it] = it + 1

             #Recharge la colormap pour les tiges ouvertes
             set_tiges_colormap()

             #print tige_id_mapper
             #Traitement des données
             ax.clear()
             ax.axis('off')
             ax.text(0.5,0.5, u"Traitement des données extraites", va='center', ha='center', color='red', transform=ax.transAxes)
             canvas.draw()

             Traite_data()


        else:
            #Add a check for black pictures
            #if check_black:
            #    files_to_process = [f.encode(sys.getfilesystemencoding()) for f in files if int(os.path.getsize(f)) >= 5000000]
            #else:
            files_to_process = [f.encode(sys.getfilesystemencoding()) for f in files]
            #print files_to_process, files
            #f.encode(sys.getfilesystemencoding())

        #Get time from pictures
        if get_photo_datetime.get():
            print('Extract time from files')
            dtphoto = []

            try:

                for f in files_to_process:
                    if os.path.exists(f):
                        tmpf = f
                    else:
                        tmpf = base_dir_path+os.path.basename(f)

                    dtphoto += [ get_photo_time(tmpf) ]

                dtphoto = array(dtphoto)
                #print dtphoto
            except Exception as e:
                dtphoto = []
                print('No time in pictures')
                print(e)

        change_button_state()
        plot_image(0, force_clear=True)

    #Try to sort images with their numbers
    #TODO sort from dtphoto !!!!
    try:
        files_to_process = sorted(files_to_process, key=lambda x: int(''.join(finddigits.findall(x.split('/')[-1]))) )
    except:
        print(u"Pas de trie des photos...car pas de numéros dans le nom des images!")

    #Restore focus on the current canvas
    canvas.get_tk_widget().focus_force()

def change_button_state():

    if len(base_tiges) > 0:
        for bt in [button_supr_all_tige, button_traiter]:
            bt.config(state=Tk.NORMAL)
    else:
        for bt in [button_supr_all_tige, button_traiter]:
            bt.config(state=Tk.DISABLED)

    if len(files_to_process) > 0:
        for bt in [button_addtige, button_listimages]:
            bt.config(state=Tk.NORMAL)
    else:
        for bt in [button_addtige, button_supr_all_tige, button_traiter, button_listimages]:
            bt.config(state=Tk.DISABLED)

    """
    if data_out != None:
        for bt in [button_export]:
            bt.config(state=Tk.NORMAL)
    else:
        for bt in [button_export]:
            bt.config(state=Tk.DISABLED)
    """

def plot_image(img_num, keep_zoom=False, force_clear=False):
    global cur_image, btige_plt, btige_text, img_object, text_object, tige_plot_object, local_photo_path, End_point_plot

    cur_image = img_num

    if force_clear:
        ax.clear()

    #Reset btige
    btige_plt = [None] * len(base_tiges)
    btige_text = [None] * len(base_tiges)


    if keep_zoom:
        oldxlims = ax.get_xlim()
        oldylims = ax.get_ylim()

    if img_num == None:
        ax.axis('off')
        ax.set_xlim(0,1)
        ax.set_ylim(0,1)
        if text_object == None:
            text_object = ax.text(0.5,0.5,"Charger des images \nou\n un fichier de resultat (rootstem_data.pkl)",ha='center',va='center')
            fig.tight_layout()
        else:
            text_object.set_text("Charger des images \nou\n un fichier de resultat (rootstem_data.pkl)")

    else:
        #ax.axis("off")
        #ax.text(0.5,0.5,'Chargement de "%s"'%files_to_process[img_num].split('/')[-1],ha='center',va='center',color='red',transform=ax.transAxes)

        #canvas.draw()
        #ax.clear()
        ax.axis('on')

        #Test d'ouverture dossier local si déplacement du dossier
        imtmp = Image(color_transform='COLOR_RGB2BGR', maxwidth=None)
        try:
            imtmp.load(files_to_process[img_num])
            local_photo_path = False
        except:
            imtmp.load(base_dir_path+files_to_process[img_num].split('/')[-1])
            local_photo_path = True


        if img_object == None:
            if imtmp.is_bw():
                img_object = ax.imshow(imtmp.render(rescale=False), cmap=mpl.cm.gray)
            else:
                img_object = ax.imshow(imtmp.render(rescale=False))
        else:
            #print shape(imtmp)
            img_object.set_data(imtmp.render(rescale=False))




        #Plot des bases de tiges
        plot_basetiges(ratio=imtmp.ratio)


        if data_out != None:

            #Plot des tiges traitée
            if tige_plot_object == None:
                tige_plot_object = []
                for ti in tige_id_mapper:
                    #print ti
                    try:
                        tmp, = ax.plot(data_out['tiges_data'].xc[ti,int(img_num),:].T*imtmp.ratio,
                                   data_out['tiges_data'].yc[ti,int(img_num),:].T*imtmp.ratio,
                                   lw=3, picker=5,label="%i"%(ti),
                                   color=tiges_colors[ti])
                        tige_plot_object += [tmp]
                    except:
                        print('No data for tige %i'%ti)
            else:
                for ti in tige_id_mapper:
                    #print ti
                    try:
                        tige_plot_object[ti].set_data(data_out['tiges_data'].xc[ti,int(img_num),:].T*imtmp.ratio, data_out['tiges_data'].yc[ti,int(img_num),:].T*imtmp.ratio)
                    except:
                        print('No data for tige %i'%ti)


        if keep_zoom:
            ax.set_xlim(oldxlims)
            ax.set_ylim(oldylims)

    canvas.draw()

########################### GESTION GRAPHIQUE DES TIGES #######################
def _addtige():
    global add_tige, nbclick, btige_plt, btige_text
    add_tige = True
    nbclick = 0
    change_button_state()


def _supr_all_tiges():
    global base_tiges, btige_plt, btige_text, tige_id_mapper
    reset_graph_data()

    #Rest base_tiges
    base_tiges = []
    btige_plt = []
    btige_text = []
    tige_id_mapper = {}
    #Replot the current image
    plot_image(cur_image, force_clear=True)
    change_button_state()

def set_tiges_colormap():
    global tiges_colors
    #Fonction pour construire le vecteur des couleurs pour les tiges
    tiges_colors = mpl.cm.Set1(linspace(0,1,len(base_tiges)+1))

def plot_basetiges(force_redraw=False, ratio=1):
    global btige_plt, btige_text, tige_id_mapper

    oldxlim = ax.get_xlim()
    oldylim = ax.get_ylim()
    to_remove = []
    for i in tige_id_mapper:
        #check tige name
        try:
            tname = tige_id_mapper[i]
            base = base_tiges[i]
        except:
            print('No base data for %i Remove it'%i)
            to_remove += [i]
            base = []

        if len(base) > 1:
            if base[0][0] < base[1][0]:
                #Vers le haut
                symb='r-'
            else:
                symb='m-'

            if btige_plt[i] == None:
                btige_plt[i], = ax.plot([base[0][0]*ratio,base[1][0]*ratio],
                                [base[0][1]*ratio,base[1][1]*ratio],
                                symb, label='%i'%i, lw=1.1, picker=5)
                #Dessin de la normale
                theta = arctan2( base[1][1]-base[0][1], base[1][0]-base[0][0] )
                L = 0.25 * sqrt( (base[1][1]-base[0][1])**2 + (base[1][0]-base[0][0])**2 ) * ratio #Taille en pixel de la normale
                xc = 0.5*(base[1][0]+base[0][0])*ratio
                yc = 0.5*(base[1][1]+base[0][1])*ratio
                xn = L * cos(theta-pi/2.)
                yn = L * sin(theta-pi/2.)
                ax.arrow(xc,yc, xn, yn, color=symb[0], length_includes_head=True, head_width=2)
            else:
                btige_plt[i].set_data([base[0][0]*ratio,base[1][0]*ratio],
                                      [base[0][1]*ratio,base[1][1]*ratio])


            if btige_text[i] == None:
                btige_text[i] = ax.text(base[0][0]*ratio,base[0][1]*ratio,'%s'%str(tname),color='r')
            else:
                btige_text[i].set_text('%s'%str(tname))
                btige_text[i].set_x(base[0][0]*ratio)
                btige_text[i].set_y(base[0][1]*ratio)

            #Add the end point if it exist
            plot_end_point(i)

        ax.set_xlim(oldxlim)
        ax.set_ylim(oldylim)


    need_save = False
    for i in to_remove:
        tige_id_mapper.pop(i)
        need_save = True

    if need_save:
        save_tige_idmapper()

    if force_redraw:
        canvas.draw()

###############################################################################

############################ OPTION POUR LE TRAITEMENT ########################
#TODO:

def control_pannel():
    pass
###############################################################################

############################ Affiche la liste des images ######################
def onChangeImage(event):
    try:
        sender=event.widget
        idx = sender.curselection()
        #print("idx %i"%idx)
        plot_image(idx[0], keep_zoom=True)
    except Exception as e:
        print(u"Erreur de chargement !!!")
        print(e)

def show_image_list():
    toplist = Tk.Toplevel(master=root)
    toplist.title("Liste des images ouvertes")


    topsbar = Tk.Scrollbar(toplist, orient=Tk.VERTICAL)
    listb = Tk.Listbox(master=toplist,  yscrollcommand=topsbar.set)
    topsbar.config(command=listb.yview)
    topsbar.pack(side=Tk.RIGHT, fill=Tk.Y)

    for i,imgpath in enumerate(files_to_process):
        listb.insert(Tk.END, "%s"%imgpath.split('/')[-1])

    listb.activate(cur_image)
    listb.bind("<<ListboxSelect>>", onChangeImage)
    listb.pack(side=Tk.LEFT, fill=Tk.BOTH, expand=1)


###############################################################################


########################## Pour le Traitement #################################
def _export_to_csv():
    #Use the function to export angle moyen et taille moyenne vers csv

    if data_out != None and len(base_tiges) > 0:
        proposed_filename = "Serie_tempotelle_AngleBout_et_taille.csv"
        outfileName = tkFileDialog.asksaveasfilename(parent=root,
                                                  filetypes=[("Comma Separated Value, csv","*.csv")],
                                                  title="Export serie temporelle",
                                                  initialfile=proposed_filename,
                                                  initialdir=base_dir_path)

    if len(outfileName) > 0:
        if len(tige_id_mapper) == 0:
            global_id=None
        else:
            global_id=[tige_id_mapper]

        #Check if I need to change image path to local path
        try:
            Image.open(files_to_process[img_num])
            change_path = lambda pname: pname['imgname']
        except:
            change_path = lambda pname: base_dir_path+pname['imgname'].split('/')[-1]

        dataout = extract_Angle_Length( traitement_file,
                             None,
                             global_tige_id=global_id,
                             image_path_mod=[change_path],
                             get_time_from_photos=get_photo_datetime.get() )


        #Add some usefull data
        Ntige = dataout.tige.unique()
        output = []
        #print Ntige
        for tige in Ntige:
            datatige = dataout.query( 'tige == %s'%str(tige) )

            #Compute dt (min)
            if get_photo_datetime.get():
                dtps = (datatige['temps']-datatige['temps'].min())/timedelta64(1,'m')
                datatige.loc[:,'dt (min)'] = dtps

            datatige.loc[:,'pictures name'] = [os.path.basename(ph) for ph in files_to_process]
            if global_id == None:
                datatige.loc[:,'x base (pix)'] = [data_out['tiges_data'].xc[tige-1,:,0].mean()]*len(datatige.index)
            else:
                if tige in tige_id_mapper.values():
                    #Besoin de recup la clef correspondant à la valeur tige dans le dictionnaire
                    tigetmp = tige_id_mapper.keys()[ tige_id_mapper.values().index(int(tige)) ]
                    datatige.loc[:,'x base (pix)'] = [data_out['tiges_data'].xc[tigetmp,:,0].mean()]*len(datatige.index)
                else:
                    datatige.loc[:,'x base (pix)'] = [data_out['tiges_data'].xc[tige-1,:,0].mean()]*len(datatige.index)

            output += [datatige]

        dataout = pd.concat(output)
        print dataout.head()
        #print datatige.tail()
        dataout.to_csv(outfileName, index=False)

def _export_xytemps_to_csv():
    #Fonction pour exporter un fichier csv contenant les tiges, temps, (xy)

    if data_out != None and len(base_tiges) > 0:
        proposed_filename = "Squelette.csv"
        outfileName = tkFileDialog.asksaveasfilename(parent=root,
                                                  filetypes=[("Comma Separated Value, csv","*.csv")],
                                                  title="Export serie temporelle",
                                                  initialfile=proposed_filename,
                                                  initialdir=base_dir_path)

    if len(outfileName) > 0:
        if len(tige_id_mapper) == 0:
            global_id=None
        else:
            global_id=[tige_id_mapper]

        #Check if I need to change image path to local path
        try:
            Image.open(files_to_process[img_num])
            change_path = lambda pname: pname['imgname']
        except:
            change_path = lambda pname: base_dir_path+pname['imgname'].split('/')[-1]

        #Get the pandas DataFrame from the data
        dataframe = extract_Angle_Length( traitement_file, outfileName, global_tige_id=global_id,
                                         image_path_mod=[change_path],
                                         get_time_from_photos=get_photo_datetime.get(),
                                         xy = True )

        dataframe.to_csv(outfileName, index=False)

def _export_mean_to_csv():
    #Use the function to export angle moyen et taille moyenne vers csv

    if data_out != None and len(base_tiges) > 0:
        proposed_filename = "Serie_tempotelle_moyenne.csv"
        outfileName = tkFileDialog.asksaveasfilename(parent=root,
                                                  filetypes=[("Comma Separated Value, csv","*.csv")],
                                                  title="Export serie temporelle",
                                                  initialfile=proposed_filename,
                                                  initialdir=base_dir_path)

    if len(outfileName) > 0:
        if len(tige_id_mapper) == 0:
            global_id=None
        else:
            global_id=[tige_id_mapper]

        #Check if I need to change image path to local path
        try:
            Image.open(files_to_process[img_num])
            change_path = lambda pname: pname['imgname']
        except:
            change_path = lambda pname: base_dir_path+pname['imgname'].split('/')[-1]

        #Get the pandas DataFrame from the data
        dataframe = extract_Angle_Length( traitement_file, None, global_tige_id=global_id,
                                         image_path_mod=[change_path],
                                         get_time_from_photos=get_photo_datetime.get() )

        #Creation de la moyenne
        datamoy = dataframe.groupby('temps').mean()
        datamoy['temps'] = dataframe.temps.unique()
        datamoy['tige'] = ['%s->%s'%(str(dataframe['tige'].min()), str(dataframe['tige'].max()))]*len(datamoy['temps'])
        #Convert to timedelta in minute
        if get_photo_datetime.get():
            dtps = (datamoy['temps']  -datamoy['temps'][0])/timedelta64(1,'m')
            datamoy['dt (min)'] = dtps

        datamoy['pictures name'] = [os.path.basename(ph) for ph in files_to_process]
        print(u"Saved to %s"%outfileName)
        datamoy.to_csv(outfileName, index=False)

def _export_meandata_for_each_tiges():
    #Fonction pour exporter un fichier par tige dans un dossier

    #Ask for a directory where to save all files
    outdir = tkFileDialog.askdirectory(title=u"Choisir un répertoire pour sauvegarder les tiges")
    if len(outdir) > 0:
        if len(tige_id_mapper) == 0:
            global_id=None
        else:
            global_id=[tige_id_mapper]

        #Check if I need to change image path to local path
        try:
            Image.open(files_to_process[img_num])
            change_path = lambda pname: pname['imgname']
        except:
            change_path = lambda pname: base_dir_path+pname['imgname'].split('/')[-1]

        #Get the pandas DataFrame from the data
        dataframe = extract_Angle_Length( traitement_file, None,
                                         global_tige_id=global_id,
                                         image_path_mod=[change_path],
                                         get_time_from_photos=get_photo_datetime.get()  )

        #Loop over tiges
        Ntige = dataframe.tige.unique()
        for tige in Ntige:
            datatige = dataframe[ dataframe.tige == tige ].copy()
            #Need to change all index
            datatige.index = arange(len(datatige))

            #Compute dt (min)
            if get_photo_datetime.get():
                dtps = (datatige['temps']  -datatige['temps'][0])/timedelta64(1,'m')
                datatige['dt (min)'] = dtps

            datatige['pictures name'] = [os.path.basename(ph) for ph in files_to_process]
            if global_id == None:
                datatige['x base (pix)'] = [data_out['tiges_data'].xc[tige-1,:,0].mean()]*len(datatige.index)
            else:
                if tige in tige_id_mapper:
                    datatige['x base (pix)'] = [data_out['tiges_data'].xc[tige_id_mapper[tige],:,0].mean()]*len(datatige.index)
                else:
                    datatige['x base (pix)'] = [data_out['tiges_data'].xc[tige-1,:,0].mean()]*len(datatige.index)

            #print datatige.head()
            outfileName = outdir+'/data_mean_for_tige_%i.csv'%tige
            print(u"Saved to %s"%outfileName)
            datatige.to_csv(outfileName, index=False)

def update_tk_progress(infos, root):
    global old_step
    if infos != None:

        im_num = infos['inum']
        old_num = infos['old_inum']
        tot = infos['tot']
        #print im_num, old_num, tot
        msg = "Traitement de l'image %i / %i"%(int(im_num),int(tot))
        root.wm_title("RootStemExtractor | %s"%msg)
        root.update_idletasks()


        #New version with Tkk progressbar
        if im_num != old_step:
            old_step = im_num
            #print(old_step, nstep, nstep-old_step)
            dstep = (im_num-old_num)/float(tot-1) * 100.
            #print(dstep)
            prog_bar.step(dstep)
            prog_bar.update_idletasks()
            root.update()

def plot_progress(**kwargs):
    global infos_traitement
    infos_traitement = kwargs

def none_print(**kwargs):
    output = kwargs

def check_process():
    global root, infos_traitement

    if thread_process.isAlive():
        update_tk_progress(infos_traitement, root)
        root.after(20, check_process)

    else:
        update_tk_progress(infos_traitement, root)
        root.wm_title("RootStemExtractor")
        thread_process.join()
        infos_traitement = None
        process_data()

def process_data():
    global data_out, imgs_out, base_pts_out, traitement_file

    #Get Queue result
    data_out = data_out.get()
    #print data_out.keys()

    #When it's done get the data from out
    imgs_out = data_out['imgs']
    base_pts_out = data_out['xypoints']
    data_out = data_out['data']


    
    #On affiche que l'on fait la sauvegarde
    ax.clear()
    ax.axis('off')
    ax.text(0.5,0.5, u"Sauvegarde des données dans ./rootstem_data.pkl", va='center', ha='center', color='red', transform=ax.transAxes)
    #tight_layout()
    canvas.draw()

    #Add pts_base to data_out
    data_out['pts_base'] = base_pts_out
    save_results(data_out, base_dir_path+"rootstem_data.pkl")
    traitement_file = base_dir_path+"rootstem_data.pkl"
    #Porcess data
    ax.clear()
    ax.axis('off')
    ax.text(0.5,0.5, u"Traitement des données extraites", va='center', ha='center', color='red', transform=ax.transAxes)
    canvas.draw()

    Traite_data()

    #Recharge la colormap pour les tiges
    set_tiges_colormap()

    #Plot the first image in the list
    plot_image(cur_image, force_clear=True)

    change_button_state()
    prog_bar.stop()

    #Restore focus on the current canvas
    canvas.get_tk_widget().focus_force()

def launch_process():
    global data_out, old_step, thread_process, Crops_data


    Crops_data = []
    max_array_size = 10000
    #Test si c'est windows -> pas de version parallele car bug
    if (platform.system().lower() == "windows") or (len(files_to_process)<2):
        #Windows as not shared memory capacity... to bad!!!
        is_thread = False
    else:
        #True
        is_thread = True

    if len(base_tiges) > 0:

        #Create a process to start process image (that can into all processors)
        #is_thread = False
        reset_graph_data()

        data_out = Queue.Queue()

        if len(files_to_process) > 1:
            print('Pre-processing last image (to get maximum plant size)')
            #Preprocessing to guess the max size of the objects and set crop zone for each of them
            pre_data = {}

            pre_data = ProcessImages(file_names=files_to_process[-1],
                                     num_images='all',
                                     num_tiges=len(base_tiges),
                                     base_points=base_tiges, thread=is_thread,
                                     pas=PAS_TRAITEMENT,
                                     tiges_seuil_offset=Tiges_seuil_offset,
                                     output_function=none_print)


            tiges_x, tiges_y, tiges_tailles, tiges_angles, tiges_lines,  = traite_tiges2(pre_data[0]['tiges_data'],
                                                                                         pas=PAS_TRAITEMENT )
            #print(tiges_tailles/0.3)
            #print(tiges_x.shape)
            max_array_size = tiges_x.shape[2] + 100

            thread_process = Thread(name="ImageProcessing",
                                              target=ProcessImages,
                                              kwargs={'file_names':files_to_process,
                                              'num_images':'all',
                                              'num_tiges':len(base_tiges),
                                              'base_points':base_tiges,
                                              'output_function':plot_progress,
                                              #'output_function_args': {'root':root},
                                              'thread':is_thread,
                                              'pas':PAS_TRAITEMENT,
                                              'outputdata':data_out,
                                              'end_points':End_point_data,
                                              'tiges_seuil_offset': Tiges_seuil_offset,
                                              'memory_size': max_array_size,
                                              'crops':Crops_data})


            thread_process.setDaemon(True)
            thread_process.start()
            #print('toto')
            check_process()

def Traite_data():
    global tiges_x, tiges_y, tiges_tailles, tiges_angles, tiges_lines, tiges_measure_zone
    tiges_x, tiges_y, tiges_tailles, tiges_angles, tiges_lines, tiges_measure_zone = traite_tiges2( data_out['tiges_data'], pas=PAS_TRAITEMENT, return_estimation_zone=True )


###############################################################################

###################### Gestion des actions du menu pour les tiges individuelles ####

def remove_tige():
    global tige_id_mapper, base_tiges, data_out
    data_out = None

    if cur_tige in tige_id_mapper:
        tname = tige_id_mapper[cur_tige]
    else:
        tname = cur_tige + 1

    print('Suppresion de la tige %i'%tname)

    base_tiges.pop(cur_tige)

    #Reconstruction de tige_id_mapper
    newidmap = {}
    if cur_tige in tige_id_mapper:
        tige_id_mapper.pop(cur_tige)
        #Changement des clefs du dico
        for i, ind in enumerate(tige_id_mapper):
            newidmap[i] = tige_id_mapper[ind]
        tige_id_mapper = newidmap
        save_tige_idmapper()

    reset_graph_data()


    plot_image(cur_image, force_clear=True, keep_zoom=True)

def _reverse_tige():
    global btige_ptl, base_tiges
    base = base_tiges[cur_tige]
    #Reverse xy
    btige_plt[cur_tige] = (base[1], base[0])

    reset_graph_data()
    plot_image(cur_image, force_clear=True, keep_zoom=True)

###################### Fenetre pour afficher une tige #########################
def export_one_tige():
    if cur_tige in tige_id_mapper:
        tname = tige_id_mapper[cur_tige]
    else:
        tname = cur_tige + 1

    proposed_filename = "tige_%i_image_%i.csv"%(tname,cur_image+1)
    outfileName = tkFileDialog.asksaveasfilename(parent=toptige,
                                              filetypes=[("Comma Separated Value, csv","*.csv")],
                                              title="Export des données tige %i"%cur_tige,
                                              initialfile=proposed_filename,
                                              initialdir=base_dir_path)
    if len(outfileName) > 0:
        #Creation du tableau avec pandas
        tx = data_out['tiges_data'].xc[cur_tige,cur_image]
        ty = data_out['tiges_data'].yc[cur_tige,cur_image]
        tsmoothx, tsmoothy, tangle, ts, tN = traite_tige2( tx, ty, data_out['tiges_data'].diam[cur_tige,cur_image]/2.0, pas=PAS_TRAITEMENT)
        tsmoothx = tsmoothx[:-1]
        tsmoothy = tsmoothy[:-1]
        tcourbure = diff(tangle)/diff(ts)
        tcourbure = hstack((tcourbure,tcourbure[-1]))
        data_tmp = {'tige':[tname]*len(tsmoothx),'image':[cur_image+1]*len(tsmoothy),
                    'angle (deg)': tangle,'x (pix)': tsmoothx,'y (pix)': tsmoothy,
                    'abscisse curviligne (pix)': ts, 'courbure c (deg/pix)':tcourbure,
                    'angle_0_360 (pix)': convert_angle(tangle)}

        data_tmp = pd.DataFrame(data_tmp, columns=['tige','image','angle (deg)','x (pix)',
                                                   'y (pix)', 'abscisse curviligne (pix)',
                                                   'courbure c (deg/pix)','angle_0_360 (pix)'])

        data_tmp.to_csv(outfileName, index=False)

def set_tige_id_mapper():
    global tige_id_mapper, Tiges_seuil_offset
    reset_graph_data()

    new_id = tktigeid.get()
    new_offset = tk_tige_offset.get()
    try:
        new_id = new_id
        tige_id_mapper[cur_tige] = new_id
        Tiges_seuil_offset[cur_tige] = new_offset
        #Save the tige_id_mapper
        save_tige_idmapper()
    except:
        print(u"Donner un nombre")


    #Replot the main image
    plot_image(cur_image, keep_zoom=True, force_clear=True)


toptige = None
tk_tige_offset = None
def show_tige_options():
    global toptige, tktigeid, tk_tige_offset

    tname = tige_id_mapper[cur_tige]
    #Ajout d'une boite tk pour l'export
    toptige = Tk.Toplevel(master=root)
    toptige.title("Tige option %i"%cur_tige)

    #Case pour changer nom de la tige
    idframe = Tk.Frame(toptige)
    Tk.Label(idframe, text='tige id:').pack(fill='x', expand=True)
    tktigeid = Tk.Entry(idframe)
    tktigeid.insert(0, str(tname))
    tktigeid.pack(fill='x', expand=True)

    Tk.Label(idframe, text='Sensibilité du seuil\n (offset n x seuil):').pack(fill='x', expand=True)
    tk_tige_offset = Tk.IntVar()

    if cur_tige in Tiges_seuil_offset:
        tk_tige_offset.set( Tiges_seuil_offset[cur_tige] )

    #label = Tk.Label(idframe, textvariable=tk_tige_offset).pack(fill='x', expand=True)
    w2 = Tk.Scale(idframe, from_=-5, to=5, resolution=0.1, variable=tk_tige_offset, orient=Tk.HORIZONTAL)
    w2.pack(fill='x', expand=True)

    Tk.Button(idframe,text="set", command = set_tige_id_mapper).pack(fill='x', expand=True)

    idframe.pack(fill='x', expand=True)

    tigebutton_export = Tk.Button(master=toptige, text='Exporter la tige vers (csv)', command=export_one_tige)
    tigebutton_export.pack(side=Tk.BOTTOM)


tktigeid = None
def show_one_tige(tige_id=None):
    global cur_tige, toptige, tktigeid, toptige

    def fit_beta(xdeb, xfin):
        global Beta_data

        #Fonction pour fitter A(t), et L(t) pour calculer b =1/sin(a(t=0)) * Da/dt/Dl/dt * R
        ideb = find(tps>=xdeb)[0]
        ifin = find(tps>=xfin)[0]

        good_tps = tps[ideb:ifin]
        good_A = A[ideb:ifin]
        good_L = tiges_tailles[cur_tige,ideb:ifin]

        #Need init time tige data
        ti_xc, ti_yc = data_out['tiges_data'].xc[cur_tige,0], data_out['tiges_data'].yc[cur_tige,0] #raw tige xc,yc
        dx, dy = diff(ti_xc,1), diff(-ti_yc,1)
        Atotti = ma.arctan2( -dx, dy )

        fitA = ma.polyfit(good_tps, good_A, 1)
        fitL = ma.polyfit(good_tps, good_L, 1)

        xfit = linspace(xdeb, xfin)
        plfitA.set_data(xfit, fitA[0]*xfit+fitA[1])

        if scale_cmpix == None:
            plfitL.set_data(xfit, fitL[0]*xfit+fitL[1] )
        else:
            plfitL.set_data(xfit, scale_cmpix*(fitL[0]*xfit+fitL[1]) )

        figt.canvas.draw_idle()

        DADT = abs(deg2rad(fitA[0]))
        DLDT = fitL[0]
        Ainit = abs(Atotti.mean()) #Angle moyen de la tige au temps 0 (deja en radiant)

        R = data_out['tiges_data'].diam[cur_tige].mean()/2.
        Runit = R
        DLDTunit = DLDT
        lscale = 'pix'
        if scale_cmpix == None:
            printR = r'$R=%0.2f$ (pix)'%R
        else:
            printR = r'$R=%0.2f$ (cm)'%(R*scale_cmpix)
            Runit *= scale_cmpix
            DLDTunit *= scale_cmpix
            lscale = 'cm'

        Betat = R/sin(Ainit) * (DADT/DLDT)
        print("R=%0.2f (%s), sin(Ainit)=%0.2f, DADT=%0.3f (rad/min), DLDT=%0.3f (%s/min), DADT/DLDT=%0.3f,  Betatilde=%0.4f"%(Runit, lscale, sin(Ainit), DADT, DLDTunit, lscale, DADT/DLDT, Betat))
        textB.set_text('$\\tilde{\\beta} = %0.4f$\n%s'%(Betat,printR))

        Beta_data[cur_tige] = {'ideb':ideb, 'ifin':ifin, 'R':Runit, 'Ainit(rad)':Ainit,'DADT(rad/min)':DADT, 'DLDT(lenght unit/min)':DLDTunit, 'Betatilde':Betat}
        #textB.set_x(xfin)

    def init_fit():
        if cur_tige in Beta_data:
            tmp = Beta_data[cur_tige]
            xdeb = tps[tmp['ideb']]
            xfin = tps[tmp['ifin']]
            try:
                fit_beta(xdeb, xfin)
                select_deb.set_xdata((xdeb,xdeb))
                select_fin.set_xdata((xfin,xfin))
            except:
                print('Error in init_fit')
                print(xdeb,ydeb)
                xdeb = tps[5]
                yfin = tps[-5]


    if tige_id!=None:
        #print(tige_id)
        cur_tige = int(tige_id)

    if cur_tige in tige_id_mapper:
        tname = tige_id_mapper[cur_tige]
    else:
        print('error')
        return None

    figt = mpl.figure('tige %s'%str(tname), figsize=(10,6))

    G = mpl.GridSpec(5, 3, wspace=.7, hspace=1)
    dtps = arange(len(tiges_tailles[cur_tige]))
    if get_photo_datetime.get() and dtphoto != []:
        tps = dtphoto
        #Temps en dt (min)
        tps = array([(t-tps[0]).total_seconds() for t in tps])/60.
        xlabel='dt (min)'
    else:
        tps = dtps
        xlabel = 'N photos'


    #print tps

    #A = convert_angle(tiges_angles[cur_tige])
    tax1 = figt.add_subplot(G[:,0])
    lcollec = LineCollection( tiges_lines[cur_tige], linewidth=(2,), color='gray')
    lcollec.set_array( dtps )
    tax1.add_collection( lcollec )
    tax1.set_xlim( (tiges_x.min(),tiges_x.max()) )
    tax1.set_ylim( (tiges_y.min(),tiges_y.max()) )
    tax1.set_xlabel('x-x0 (pix)')
    tax1.set_ylabel('y-y0 (pix)')
    tax1.axis('equal')

    #Affiche les zones de calcul de l'angle moyen
    #colors = cm.jet(linspace(0,1,len(tps)))
    for t in [0]:
        xt = tiges_x[cur_tige, t, ~tiges_x[cur_tige,t].mask]
        yt = tiges_y[cur_tige, t, ~tiges_y[cur_tige,t].mask]
        try:
            xlims = [xt[tiges_measure_zone[cur_tige][t][0]], xt[tiges_measure_zone[cur_tige][t][1]]]
            ylims = [yt[tiges_measure_zone[cur_tige][t][0]], yt[tiges_measure_zone[cur_tige][t][1]]]
            colortige, = tax1.plot(xt,yt,'k',lw=2.5)
            lims, = tax1.plot(xlims,ylims,'o',color='m',ms=10)
        except:
            pass


    tax2 = figt.add_subplot(G[3:,1:])
    tax3 = figt.add_subplot(G[1:3,1:], sharex=tax2)
    tax4 = figt.add_subplot(G[0,1:], sharex=tax2)
    #Affiche les timeseries Angle au bout et Taille
    if len(tps) > 1:
        #tax2 = figt.add_subplot(G[2:,1:])
        if scale_cmpix == None:
            tax2.plot(tps, tiges_tailles[cur_tige], '+-', color=tiges_colors[cur_tige], lw=2 )
            tax2.set_ylabel('Taille (pix)')
        else:
            tax2.plot(tps, tiges_tailles[cur_tige]*scale_cmpix, '+-', color=tiges_colors[cur_tige], lw=2 )
            tax2.set_ylabel('Taille (cm)')

        plfitL, = tax2.plot([],[],'m', lw=1.5)
        tax2.set_xlabel(xlabel)
        l1, = tax2.plot([tps[0],tps[0]],tax2.get_ylim(),'k--',lw=1.5)

        #tax3 = figt.add_subplot(G[:2,1:],sharex=tax2)
        if angle_0_360.get():
            A = convert_angle(tiges_angles[cur_tige])
        else:
            A = tiges_angles[cur_tige]

        #A = tiges_angles[tige_id]
        tax3.plot(tps, A, '+-',  color=tiges_colors[cur_tige], lw=2 )
        tax3.set_ylabel('Tip angle (deg)')
        tax3.set_xlabel(xlabel)
        l2, = tax3.plot([tps[0],tps[0]],tax3.get_ylim(),'k--',lw=1.5)
        try:
            select_deb, = tax4.plot([tps[5]]*2, [0,1],'m', lw=3, picker=5)
            select_fin, = tax4.plot([tps[-5]]*2, [0,1],'m', lw=3, picker=5)
        except:
            print('Not enouth images for estimation of beta')
            select_deb, = tax4.plot([tps[0]]*2, [0,1],'m', lw=3, picker=5)
            select_fin, = tax4.plot([tps[0]]*2, [0,1],'m', lw=3, picker=5)
            
        plfitA, = tax3.plot([],[],'m', lw=1.5)
        dlines = DraggableLines([select_deb, select_fin])
        #tax4.set_xticks([])
        tax4.set_yticks([])
        textB = tax4.text(tps[-2], 0.5, '', va='center', ha='right')
        init_fit()
    else:
        #Si une seul image est traité on montre Angle(f(s)) et Courbure(f(s))
        tsmoothx, tsmoothy, tangle, ts, tN = traite_tige2( data_out['tiges_data'].xc[cur_tige,0], data_out['tiges_data'].yc[cur_tige,0], 
                                                          data_out['tiges_data'].diam[cur_tige,0]/2.0, pas=PAS_TRAITEMENT)
        #Recentre tout entre 0 et 2pi (le 0 est verticale et rotation anti-horraire)
        #tangle[tangle<0] += 2*pi
        tangle = rad2deg(tangle)
        if angle_0_360.get():
            tangle = convert_angle(tangle)



        if scale_cmpix == None:
            tax2.plot( ts, tangle, color=tiges_colors[cur_tige], lw=2 )
            tax3.plot( ts[:-5], diff(tangle[:-4])/diff(ts[:-4]), color=tiges_colors[cur_tige], lw=2 )
            tax3.set_xlabel('Abscice curviligne, s (pix)')
            tax3.set_ylabel('Courbure (deg/pix)')
        else:
            tax2.plot( ts*scale_cmpix, tangle, color=tiges_colors[cur_tige], lw=2 )
            tax3.plot( ts[:-5]*scale_cmpix, diff(tangle[:-4])/diff(ts[:-4]*scale_cmpix), color=tiges_colors[cur_tige], lw=2 )
            tax3.set_xlabel('Abscice curviligne, s (cm)')
            tax3.set_ylabel('Courbure (deg/cm)')

        tax2.set_ylabel('Angle (deg)')
        tax4.axis('off')

    #figt.tight_layout()

    def OnPick(e):
        dlines.on_press(e)

    def OnMotion(e):
        dlines.on_motion(e)

    def OnRelease(e):
        dlines.on_release(e)

        if tax4.contains(e)[0]:
            #Update fillbetween_x
            xa = select_deb.get_xdata()[0]
            xb = select_fin.get_xdata()[0]
            xdeb = min(xa,xb)
            xfin = max(xa,xb)
            #print(xdeb, xfin)
            fit_beta(xdeb, xfin)

    def OnClick(event):

        if event.xdata is not None and (tax3.contains(event)[0] or tax2.contains(event)[0]):

            t = int(round(event.xdata))
            #Min distance to clicked point
            t = ((tps - event.xdata)**2).argmin()
            if len(tps) > 1:
                xt = tiges_x[cur_tige, t, ~tiges_x[cur_tige,t].mask]
                yt = tiges_y[cur_tige, t, ~tiges_y[cur_tige,t].mask]
                try:
                    xlims = [xt[tiges_measure_zone[cur_tige][t][0]], xt[tiges_measure_zone[cur_tige][t][1]]]
                    ylims = [yt[tiges_measure_zone[cur_tige][t][0]], yt[tiges_measure_zone[cur_tige][t][1]]]
                    colortige.set_data(xt,yt)
                    lims.set_data(xlims,ylims)
                except:
                    pass

                l1.set_xdata([tps[t],tps[t]])
                l2.set_xdata([tps[t],tps[t]])


                #redraw figure
                figt.canvas.draw()

                #Change figure on main frame
                plot_image(t, keep_zoom=True)

    def on_close(event):
        global toptige
        #Gestion fermeture de la figure
        #Export des données
        save_tige_idmapper()
        try:
            toptige.destroy()
            toptige = None
        except:
            pass

        mpl.close('tige %s'%str(tname))

    #Add click event for the figure
    figt.canvas.mpl_connect('button_press_event', OnClick)
    figt.canvas.mpl_connect('close_event', on_close)
    figt.canvas.mpl_connect('pick_event', OnPick)
    figt.canvas.mpl_connect('button_release_event', OnRelease)
    figt.canvas.mpl_connect('motion_notify_event', OnMotion)

    #Show the figure
    figt.show()

def show_B(tige_id=None):

    if local_photo_path:
        imgs_sf = ( files_to_process[0].split('/')[-1], files_to_process[-1].split('/')[-1] )
    else:
        imgs_sf = ( files_to_process[0], files_to_process[-1] )
    #Test growth zone

    get_growth_length(data_out['tiges_data'], cur_tige, imgs=imgs_sf)
    mpl.gcf().show()


    #mpl.close('Growth lenght')

def plot_moyenne():
    #Fonction pour tracer les courbes moyennes des tiges
    #mpl.close('all')

    if len(files_to_process) > 1:
        #On regroupe les données dans un tableau pandas
        #Test si on doit mapper les ids des tiges
        if len(tige_id_mapper) == 0:
            return None
            #global_id=None
            #legendtige = ["tige %i"%(i+1) for i in arange(0,len(base_tiges))]
        else:
            legendtige=["tige %s"%str(tige_id_mapper[i]) for i in tige_id_mapper]
            global_id=[tige_id_mapper]

        #Check if I need to change image path to local path
        try:
            Image.open(files_to_process[img_num])
            change_path = lambda pname: pname['imgname']
        except:
            change_path = lambda pname: base_dir_path+pname['imgname'].split('/')[-1]

        dataframe = extract_Angle_Length( traitement_file, None, global_tige_id=global_id,
                                         image_path_mod=[change_path], get_time_from_photos=get_photo_datetime.get() )


        fig = mpl.figure(u'Série temporelles avec moyenne',figsize=(10,5))
        G = mpl.GridSpec(2,4)
        ax1 = mpl.subplot(G[0,:3])
        if angle_0_360.get():
            dataa = "angle_0_360"
        else:
            dataa = "angle"


        plot_sequence(dataframe, tige_color=None, show_lims=False, ydata=dataa, tige_alpha=1)
        if not get_photo_datetime.get():
            mpl.xlabel("Number of images")
        mpl.grid()

        mpl.subplot(G[1,:3], sharex=ax1)
        yplot = 'taille'
        if scale_cmpix != None:
            dataframe['taille(cm)'] = dataframe['taille'] * scale_cmpix
            yplot = 'taille(cm)'
        plot_sequence(dataframe, tige_color=None, show_lims=False, ydata=yplot, tige_alpha=1)
        ylims = mpl.ylim()
        mpl.ylim(ylims[0]-20, ylims[1]+20)
        if not get_photo_datetime.get():
            mpl.xlabel("Number of images")
        mpl.grid()

        #axl = subplot(G[:,3])
        #print legendtige
        ax1.legend(legendtige+["Moyenne"], bbox_to_anchor=(1.02, 1), loc=2)

        fig.tight_layout()
        fig.show()

    else:
        print(u"Pas de serie temporelle, une seule photo traitée")


#Fonction pour tester la detection sur une tige et une image
def test_detection():

    #Au moins une base de tige et une image
    if cur_image is not None and len(base_tiges)>0 :

        #Load cur image
        imtmp = Image(color_transform='COLOR_RGB2BGR', maxwidth=None)
        imtmp.load(files_to_process[cur_image])

        oldxlims = ax.get_xlim()
        oldylims = ax.get_ylim()

        #Affiche l'image
        fig = mpl.figure("test")
        axt = fig.add_subplot(111)
        #Check BW
        if imtmp.is_bw():
            axt.imshow(imtmp.render(rescale=False), cmap=mpl.cm.gray)
        else:
            axt.imshow(imtmp.render(rescale=False))

        axt.set_xlim(oldxlims)
        axt.set_ylim(oldylims)

        fig.show()

        #Selection de la base la plus proche du centre de l'image
        xmbases = array([0.5*(b[0][0]+b[1][0]) for b in base_tiges])
        ymbases = array([0.5*(b[0][1]+b[1][1]) for b in base_tiges])
        good_tige = argmin( sqrt( (xmbases-mean(oldxlims))**2 + (ymbases-mean(oldylims))**2 ) )

        #print("Test sur la tige %i"%good_tige)
        #Lance la detection avec affichage des traits de coupe
        Process_images( [files_to_process[cur_image]],'all', 1, base_points = [ base_tiges[good_tige] ], pas = PAS_TRAITEMENT, thread=False, show_tige=True)

def save_tige_idmapper():
    print('Save data...')
    #Save the tige_id_mapper
    with open(base_dir_path+'tige_id_map.pkl','wb') as f:
        pkl.dump({'tige_id_mapper':tige_id_mapper,
                  'scale_cmpix':scale_cmpix,
                  'B_data': B_data,
                  'Beta_data': Beta_data,
                  'End_point_data': End_point_data,
                  'Tiges_seuil_offset': Tiges_seuil_offset}, f)

#Calcul de la longueur de croissance
GoodL = 0 #Some global variables
Lc = 0
Lcexp = 0
B = 0
Bexp = 0
def get_growth_length(tiges, cur_tige, thresold='auto', imgs = None, pas = 0.3):
    """
    Compute the curvature length as described in the AC model of Bastien et al.

    tiges: is the tiges instance
    thresold[auto]: the thresold if computed as 2 times the mean diameter.
    imgs[None]: can be a list of two images to plot the initial and final position of the organ
    """


    if cur_tige in B_data:
        save_B_data = B_data[cur_tige]
        cur_tps = save_B_data['num_img_fit'] #Temps (ou photo) sur laquel on prend la forme finale
    else:
        cur_tps = -1
        save_B_data = None  #Temps (ou photo) sur laquel on prend la forme finale

    print(cur_tps)

    def compute_R(model, data):
        #Cf wikipedia Coefficient_of_determination

        sstot = ma.sum( (data - data.mean())**2 )
        #ssreg = ma.sum( (model - data.mean())**2 )
        ssres = ma.sum( (model - data)**2 )

        R = 1 - ssres/sstot
        return R

    def Curve_zone(ds_start, Arad, cur_tps):

        s = cumsum( sqrt( sdx[cur_tps]**2 + sdy[cur_tps]**2 ) )
        if scale_cmpix != None:
            s *= scale_cmpix

        print(ds_start, cur_tps, s)
        Sc = s[ds_start:]-s[ds_start]
        AA0 = Arad[cur_tps, ds_start:]/Arad[cur_tps, ds_start]
        signal = ma.log(AA0[:len(Sc[~AA0.mask])])

        pl_A_exp.set_data( Sc[~signal.mask], AA0[~signal.mask] )
        pl_A_log.set_data( Sc[~signal.mask], signal[~signal.mask] )

        try:
            ax4.set_xlim(0, Sc.max())
            ax4.set_ylim(AA0.min(), AA0.max())
            ax5.set_xlim(0, Sc.max())
            ax5.set_ylim(signal.min(), signal.max())
        except:
            pass


        return Sc, AA0, signal


    def fit_As(Sc, signal, cur_tps):
        global GoodL, Lc, Lcexp, B, Bexp
        try:
            min_func = lambda p, x, y: sum( sqrt( (x*p[0] - y)**2 ) )
            min_func_exp = lambda p, x, y: sum( sqrt( (ma.exp(-x/p[0]) - y)**2 ) )

            #p0 = signal.std()/Sc[~signal.mask].std()
            #print(p0)

            opt_params = fmin(min_func, [1.0], args = (Sc[~signal.mask], signal[~signal.mask]))
            opt_params_exp = fmin(min_func_exp, [1.0], args = (Sc[~signal.mask], ma.exp(signal[~signal.mask])))
            #fitA = poly1d( ma.polyfit(Sc, log(AA0), 1) )
            #print(opt_params)
            #print(opt_params_exp)
            #print(fitA)

            Lc = -1/opt_params[0]
            Lgz = Sc.max()



            Si = sinit-sinit[0]
            Lo = Si.max()
            GoodL = min((Lgz,Lo))
            B = GoodL/Lc
            Lcexp = opt_params_exp[0]
            Bexp = GoodL/Lcexp
            if cur_tps == -1:
                print_tps = dx.shape[0]
            else:
                print_tps = cur_tps

            length_scale='pix'
            if scale_cmpix != None:
                length_scale = 'cm'

            text_infos.set_text("Img: %i, unit: %s, Lzc=%0.2f, Ltot=%0.2f || fit (A/A0): Lc=%0.2f, B=%0.2f || fit log(A/A0): Lc=%0.2f, B=%0.2f"%(print_tps, length_scale, Lgz, Lo, Lcexp, Bexp, Lc, B))

            xtmp = linspace(0,max(mpl.gca().get_xlim()))
            fit4log.set_data(xtmp, exp(-xtmp/Lc))
            fit4exp.set_data(xtmp, exp(-xtmp/opt_params_exp[0]))

            Rlogexp = compute_R(ma.exp(-Sc[~signal.mask]/Lc), ma.exp(signal[~signal.mask]))
            Rexp = compute_R(ma.exp(-Sc[~signal.mask]/opt_params_exp[0]), ma.exp(signal[~signal.mask]))
            fit4log.set_label('$R^2 = %0.3f$'%Rlogexp)
            fit4exp.set_label('$R^2 = %0.3f$'%Rexp)
            ax4.legend(loc=0,prop={'size':10})

            xtmp = linspace(0,max(mpl.gca().get_xlim()))
            fit5.set_data(xtmp, -xtmp/Lc)

            Rlog = compute_R(-Sc[~signal.mask]/Lc, signal[~signal.mask])
            fit5.set_label('$R^2 = %0.3f$'%Rlog)
            ax5.legend(loc=0,prop={'size':10})

        except:
            print('Fit Error!')


    def OnPick(evt):
        #Connect dragable lines
        dlines.on_press(evt)

    def OnMotion(evt):
        dlines.on_motion(evt)

    def On_close(evt):
        global B_data
        #Do things when close the B windows.

        #Save the data to tige_id_map.pkl
        is_start = find(sinit >= pl_seuil_tps.get_data()[0][0])[0]
        icur_img = int(pl_curv_img.get_data()[1][0])

        #Check if it is possible
        if icur_img >= dx.shape[0]-1:
            icur_img = dx.shape[0]-1

        unit='pix'
        if scale_cmpix != None:
            unit = 'cm'
        B_data[cur_tige]={'s0': is_start, 'num_img_fit': icur_img, 'Lgz': GoodL,
                          'Lc (fit Log(A/A0))':Lc, 'Lc (fit exp(A/A0))': Lcexp,
                          'unit': unit, 'B (log)': B, 'B (exp)': Bexp}

        save_tige_idmapper()

        print(B_data[cur_tige])


    def OnRelease(evt):


        dlines.on_release(evt)

        if dlines.changed:
            dlines.changed = False
            #Update final position of both lines
            #pl_seuil_tps.set_xdata([evt.xdata]*2)
            #pl_seuil_tps2.set_xdata([evt.xdata]*2)

            cur_tps = int(pl_curv_img.get_data()[1][0])
            if cur_tps >= dx.shape[0]:
                cur_tps = -1


            ds_start = find(sinit >= pl_seuil_tps.get_data()[0][0])[0]
            pl_seuil_pts.set_data(tiges.xc[cur_tige,0,ds_start]-imgxmi, tiges.yc[cur_tige,0,ds_start]-imgymi)

            Sc, AA0, signal = Curve_zone(ds_start, A, cur_tps)
            fit_As(Sc, signal, cur_tps)


            scur = cumsum( sqrt( sdx[cur_tps]**2 + sdy[cur_tps]**2 ) )

            if scale_cmpix != None:
                scur *= scale_cmpix

            if cur_tps == -1:
                print_tps = dx.shape[0]-1
            else:
                print_tps = cur_tps

            pl_average.set_data(scur[~A[cur_tps,:].mask], A[cur_tps,~A[cur_tps,:].mask]-A[cur_tps,0])
            pl_average.set_label('Img %i'%print_tps)
            ax2.legend(loc=0,prop={'size':10})
            pl_photo_cur_tige.set_data(tiges.xc[cur_tige,cur_tps]-imgxmi,
                                       tiges.yc[cur_tige,cur_tps]-imgymi)
            fig.canvas.draw()

    #print(cur_tige)

    #Need init time tige data
    ti_xc, ti_yc = tiges.xc[cur_tige,0], tiges.yc[cur_tige,0] #raw tige xc,yc
    #print(diff(ti_s)[::20])

    #Need last time tige data
    tf_xc, tf_yc = tiges.xc[cur_tige,-1], tiges.yc[cur_tige,-1]
    #print(diff(tf_s)[::20])

    xmax = max((ti_xc.max(), tf_xc.max()))
    xmin = min((ti_xc.min(), tf_xc.min()))
    ymax = max((ti_yc.max(), tf_yc.max()))
    ymin = min((ti_yc.min(), tf_yc.min()))

    imgxma = int(xmax+0.02*xmax)
    imgxmi = int(xmin-0.02*xmin)
    imgyma = int(ymax+0.02*ymax)
    imgymi = int(ymin-0.02*ymin)

    #print((imgxmi, imgymi, imgxma-imgxmi, imgyma-imgymi))
    fig = mpl.figure('B for organ %i'%(cur_tige), figsize=(12,10))


    G = mpl.GridSpec(3,4)
    ax1 = mpl.subplot(G[:2,2:])
    if imgs != None:
        tmpi = Image(use_bw=True)
        tmpf = Image(use_bw=True)
        tmpi.load(imgs[0])
        tmpf.load(imgs[-1])
        imgi = tmpi.render()[imgymi:imgyma, imgxmi:imgxma]
        imgf = tmpf.render()[imgymi:imgyma, imgxmi:imgxma]

        ax1.imshow(imgi, 'gray')
        ax1.imshow(imgf, 'gray', alpha=0.5)

    ax1.plot(tiges.xc[cur_tige,0]-imgxmi, tiges.yc[cur_tige,0]-imgymi, 'g-', lw=2)
    pl_photo_cur_tige, = ax1.plot(tiges.xc[cur_tige,cur_tps]-imgxmi, tiges.yc[cur_tige,cur_tps]-imgymi, 'm--', lw=2)
    ax1.axis('equal')
    ax1.axis('off')
    #subplot(G[:2,0])
    #plot(ti_s, rad2deg(ti_angle))
    #plot(tf_s, rad2deg(tf_angle))

    #Caclul du spatio-temporel
    coupe_end = 5
    xt, yt = tiges.xc[cur_tige,:,:-coupe_end], tiges.yc[cur_tige,:,:-coupe_end]
    lx, ly = xt.shape
    dx, dy = diff(xt,1), diff(-yt,1)

    sdx, sdy = zeros_like(dx) - 3000, zeros_like(dy) - 3000
    W = int( round( (tiges.diam[cur_tige].mean()/2.)/pas ) * 2.0 )

    wind = ones( W, 'd' )
    for i in xrange(dx.shape[0]):
        dxT = ma.hstack( [ dx[i, W-1:0:-1], dx[i,:], dx[i,-1:-W:-1] ] )
        dyT = ma.hstack( [ dy[i, W-1:0:-1], dy[i,:], dy[i,-1:-W:-1] ] )
        cx=convolve(wind/wind.sum(), dxT, mode='valid')[(W/2-1):-(W/2)-1]
        cy=convolve(wind/wind.sum(), dyT, mode='valid')[(W/2-1):-(W/2)-1]
        sdx[i,:len(cx)] = cx
        sdy[i,:len(cy)] = cy


    sdx = ma.masked_less_equal(sdx, -100.0)
    sdy = ma.masked_less_equal(sdy, -100.0)
    Arad = ma.arctan2( -sdx, sdy )
    A = rad2deg( Arad )


    sinit = cumsum( sqrt( sdx[0]**2 + sdy[0]**2 ) )
    sfinal = cumsum( sqrt( sdx[-1]**2 + sdy[-1]**2 ) )
    scur = cumsum( sqrt( sdx[cur_tps]**2 + sdy[cur_tps]**2 ) )

    lscale = 'pix'
    if scale_cmpix != None:
        sinit *= scale_cmpix
        scur *= scale_cmpix
        sfinal *= scale_cmpix
        lscale = 'cm'

    ax2 = mpl.subplot(G[0,:2])
    if cur_tps == -1:
        print_tps = dx.shape[0] - 1
    else:
        print_tps = cur_tps

    ax2.plot(scur[~A[-1,:].mask], A[-1,~A[-1,:].mask]-A[-1,0], 'gray', label='Last')
    pl_average, = ax2.plot(scur, A[cur_tps,:]-A[cur_tps,0], 'm', label='Img %i'%print_tps)
    pl_first, = ax2.plot(sinit, A[0,:]-A[0,0], 'g', label='First')
    seuil = (A[0,:len(A[0][~A[0].mask])/2]-A[0,0]).mean()
    xlims = ax2.get_xlim()
    pl_seuil_A, = ax2.plot(xlims,[seuil]*2, 'r--')
    mpl.xlabel('s (%s)'%lscale)
    mpl.ylabel('Angle A-A(s=0) (deg)')

    #Find intersection with an offset
    if save_B_data != None:
        ds_start = save_B_data['s0']
    else:
        try:
            ds_start = len(A.mean(0)) - find( abs( (A.mean(0)-A.mean(0)[0]) - seuil )[::-1] < 5.0 )[0]
        except:
            ds_start = 0

    #check if the value is possible otherwise put the value to the half of s
    if ds_start >= len(sinit[~sinit.mask]):
        ds_start = int(len(sinit[~sinit.mask])/2)



    pl_seuil_tps, = ax2.plot([sinit[ds_start]]*2, ax2.get_ylim(), 'k--', picker=10)
    pl_seuil_pts, = ax1.plot(tiges.xc[cur_tige,0,ds_start]-imgxmi, tiges.yc[cur_tige,0,ds_start]-imgymi, 'ro', ms=12)
    ax2.legend(loc=0,prop={'size':10})

    ax3 = mpl.subplot(G[1,:2], sharex=ax2)
    colorm = ax3.pcolormesh(sfinal, arange(dx.shape[0]), A)
    ax3.set_ylim(0, dx.shape[0])
    cbar = mpl.colorbar(colorm, use_gridspec=True, pad=0.01)
    cbar.ax.tick_params(labelsize=10)
    pl_seuil_tps2, = ax3.plot([sinit[ds_start]]*2, ax3.get_ylim(), 'k--', picker=10)
    if cur_tps == -1:
        tmpy = dx.shape[0] - 1
    else:
        tmpy = cur_tps

    pl_curv_img, = ax3.plot(ax3.get_xlim(), [tmpy]*2, 'm--', picker=10)
    mpl.ylabel('Num Photo')
    mpl.xlabel('s (%s)'%lscale)

    #Fit sur le log(A/A0)
    ax4 = mpl.subplot(G[2,:2])
    pl_A_exp,= ax4.plot([], [], 'o')
    fit4exp, = ax4.plot([], [], 'r', lw=2, label='R')
    fit4log, = ax4.plot([], [], 'g--', lw=2, label='R')
    mpl.ylabel('A/A0')
    mpl.xlabel('Sc (%s)'%lscale)

    ax5 = mpl.subplot(G[2,2:])
    pl_A_log, = ax5.plot([], [], 'o')
    fit5, = ax5.plot([], [], 'g', lw=2, label='R')
    mpl.ylabel('log(A/A0)')
    mpl.xlabel('Sc (%s)'%lscale)

    text_infos = mpl.figtext(.5,0.01,'', fontsize=11, ha='center', color='Red')
    #Plot the growth zone (i.e. the one that curve)
    Sc, AA0, signal = Curve_zone(ds_start, A, cur_tps)


    #AA0 = Arad[-1,ds_start:]

    fit_As(Sc, signal, cur_tps)


    dlines = DraggableLines([pl_seuil_tps,pl_seuil_tps2, pl_curv_img],
                            linked=[pl_seuil_tps,pl_seuil_tps2])
    fig.canvas.mpl_connect('pick_event', OnPick)
    fig.canvas.mpl_connect('button_release_event', OnRelease)
    fig.canvas.mpl_connect('motion_notify_event', OnMotion)
    fig.canvas.mpl_connect('close_event', On_close)

    mpl.tight_layout()


def measure_pixels():
    global add_dist_draw, dist_measure_pts, plt_measure
    #Function to measure the pixel by drawing two points
    if not add_dist_draw:
        add_dist_draw = True
        dist_measure_pts = [] #empty points list to store them
        try:
            plt_measure.set_data([],[])
            canvas.draw()
        except:
            pass


def update_scale_pxcm():
    global scale_cmpix, pixel_distance, cm_distance

    done = False
    try:
        px = float(pixel_distance.get())
        cm = float(cm_distance.get())
        print("scale %0.4f cm/pix"%(cm/px))
        scale_cmpix = cm/px
        save_tige_idmapper()
        done = True
    except:
        print('Error in scale')
        scale_cmpix = None #reset scale to none

    if done:
        pixel_distance.delete(0, Tk.END)
        cm_distance.delete(0, Tk.END)
        pixel_distance.insert(0, '1.0')
        cm_distance.insert(0, '%0.4f'%scale_cmpix)
        try:
            plt_measure.set_data([],[])
            canvas.draw()
        except:
            pass

def pixel_calibration():
    #Function to calibrate pixel->cm
    global pixel_distance, cm_distance

    def update_cm_distance(sv):
        #new_val = sv.get()
        update_scale_pxcm()

    #Ajout d'une boite tk pour l'export
    topcal = Tk.Toplevel(master=root)
    topcal.title("Calibration pix->cm")

    #Case pour changer nom de la tige
    calframe = Frame(topcal)
    pixel_distance = Entry(calframe, width=10)
    if scale_cmpix != None:
        pixel_distance.insert(0, str('1.0'))


    cm_distance = Entry(calframe, width=10)
    if scale_cmpix != None:
        cm_distance.insert(0, str('%0.4f'%scale_cmpix))

    Tk.Label(calframe, text='pixel:').pack()
    pixel_distance.pack()
    Tk.Label(calframe, text='cm:').pack()
    cm_distance.pack()
    calframe.pack()

    calbutton_calibration = Button(master=topcal, text='Measure distance',
                                   command=measure_pixels)
    calbutton_calibration.pack(fill=Tk.X)

    calbutton_updatecalibration = Button(master=topcal, text='Update scale',
                                         command=update_scale_pxcm)
    calbutton_updatecalibration.pack(fill=Tk.X)


def _export_tige_id_mapper_to_csv():
    #Function to export

    proposed_filename = "Phenotype_gravi_proprio.csv"
    outfileName = tkFileDialog.asksaveasfilename(parent=root,
                                                  filetypes=[("Comma Separated Value, csv","*.csv")],
                                                  title="Export serie temporelle",
                                                  initialfile=proposed_filename,
                                                  initialdir=base_dir_path)

    if len(outfileName) > 0:
        ntiges = len(base_tiges)
        output = pd.DataFrame()
        try:
            for i in range(ntiges):
                tmp_out = {}
                tmp_out['scale (cm/pix)'] = scale_cmpix
                tmp_out['id_tige'] = i
                if i in tige_id_mapper:
                    tmp_out['tige_name'] = tige_id_mapper[i]
                else:
                    tmp_out['tige_name'] = i

                if i in B_data:
                    for key in B_data[i]:
                        tmp_out[key] = B_data[i][key]

                if i in Beta_data:
                    for key in Beta_data[i]:
                        tmp_out[key] = Beta_data[i][key]

                output = output.append(tmp_out, ignore_index=True)
        except Exception as e:
                print('Error in tige_id_mapper data format\nError (tige %i): %s'%(i, e))

        print(u"Saved to %s"%outfileName)
        output.to_csv(outfileName, index=False)



def plot_end_point(tige_id):
    global End_point_plot

    if tige_id in End_point_data:

        xc = End_point_data[tige_id]['xc']
        yc = End_point_data[tige_id]['yc']
        R = End_point_data[tige_id]['R']
        teta = linspace(0,2*pi)
        xcircle = float(xc[cur_image]) + float(R)*cos(teta)
        ycircle = float(yc[cur_image]) + float(R)*sin(teta)

        if tige_id in End_point_plot:
            #print(End_point_plot[tige_id])
            End_point_plot[tige_id]['full_path_plot'].set_data(xc, yc)
            End_point_plot[tige_id]['cur_image_plot'].set_data(xc[cur_image], yc[cur_image])

            End_point_plot[tige_id]['cur_image_circle'].set_data(xcircle, ycircle)
        else:
            pl1, = ax.plot(xc,yc,'w+-',alpha=0.4)
            pl2, = ax.plot(xc[cur_image], yc[cur_image], 'ro', ms=10)
            pl3, = ax.plot(xcircle, ycircle, 'r--')
            End_point_plot[tige_id] = {'full_path_plot': pl1, 'cur_image_plot': pl2,
                                        'cur_image_circle': pl3}


def process_end_marker():
    global root, infos_traitement, End_point_data

    if thread_process.isAlive():
        update_tk_progress(infos_traitement, root)
        root.after(20, process_end_marker)

    else:
        update_tk_progress(infos_traitement, root)
        root.wm_title("RootStemExtractor")
        thread_process.join()
        infos_traitement = None

        #Get Queue result
        data_tmp = output_marker.get()

        End_point_data[cur_tige]["xc"] = data_tmp['xc']
        End_point_data[cur_tige]["yc"] = data_tmp['yc']

        #Save results
        save_tige_idmapper()

        #Draw things
        plot_end_point(cur_tige)
        canvas.draw()

output_marker = None
def add_end_marker():

    def call_back_square(eclick, erelease):
        global xmin, xmax, ymin, ymax, End_point_data, thread_process, output_marker

        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata


        xmin, xmax = x1, x2
        ymin, ymax = y1, y2
        xm, ym = mean((xmin,xmax)), mean((ymin,ymax))

        #ax.plot( (xmin,xmax,xmax,xmin,xmin), (ymin,ymin,ymax,ymax,ymin),'r--')
        #ax.plot(xm,ym,'mo')
        Rs.set_active(False)

        #Definition du pattern
        imgt = Image(use_bw=True)
        imgt.load(files_to_process[cur_image])

        pattern = imgt.render(rescale=False)[ymin:ymax, xmin:xmax]
        pattern_center = (xm, ym)

        radius = max(pattern.shape)
        End_point_data[cur_tige] = {'R':radius/2.0}


        if thread_process == None or thread_process.isAlive() == False:
            output_marker = Queue.Queue()

            thread_process = Thread(name="PatternProcessing",
                                              target=compute_pattern_motion,
                                              kwargs={'images_names':files_to_process,
                                              'pattern': pattern,
                                              'pattern_center':pattern_center,
                                              'output_data':output_marker,
                                              'output_function':plot_progress})



            thread_process.setDaemon(True)
            thread_process.start()
            #print('toto')
            process_end_marker()

        #Run the pattern detection on all pictures
        #xc, yc = compute_pattern_motion(files_to_process, pattern, pattern_center)

        #Ask to put 4 points


    #Transform in points to maximum square
    Rs = RectangleSelector(ax, call_back_square,
                       drawtype='line', useblit=True,
                       button=[1, 3],  # don't use middle button
                       minspanx=10, minspany=10,
                       spancoords='pixels',
                       interactive=True)
    Rs.set_active(True)

def remove_end_point():
    global End_point_data, End_point_plot

    End_point_plot[cur_tige]['full_path_plot'].remove()
    End_point_plot[cur_tige]['cur_image_plot'].remove()
    End_point_plot[cur_tige]['cur_image_circle'].remove()
    canvas.draw_idle()
    canvas.draw()

    End_point_data.pop(cur_tige)
    End_point_plot.pop(cur_tige)
    save_tige_idmapper()


###############################################################################

################################ Main windows #################################
if __name__ == '__main__':

    root = Tk.Tk()
    root.style = Style()
    root.style.theme_use("clam")
    root.wm_title("RootStemExtractor -- version:%s" % (__version__))
    print("RootStemExtractor -- version:%s" % (__version__))
    
    #TOP MENU BAR
    menubar = Tk.Menu(root)

    # Export menu
    exportmenu = Tk.Menu(menubar, tearoff=0)
    exportmenu.add_command(label=u"Série temporelle moyenne", command=_export_mean_to_csv)
    exportmenu.add_command(label=u"Séries temporelles par tiges", command=_export_meandata_for_each_tiges)
    exportmenu.add_command(label=u"Séries temporelles globales [A(t,s=tip), L(t)]", command=_export_to_csv)
    exportmenu.add_command(label=u"Séries temporelles globales + squelette", command=_export_xytemps_to_csv)
    exportmenu.add_command(label=u"Phenotype (graviception, proprioception)", command=_export_tige_id_mapper_to_csv)
    menubar.add_cascade(label="Exporter", menu=exportmenu)

    #Plot menu
    plotmenu = Tk.Menu(menubar, tearoff=0)
    plotmenu.add_command(label=u"Série temporelles", command=plot_moyenne)
    menubar.add_cascade(label="Figures", menu=plotmenu)

    #Options menu
    options_menu = Tk.Menu(menubar)
    #Pour chercher le temps dans les données EXIFS des images
    get_photo_datetime = Tk.BooleanVar()
    get_photo_datetime.set(True)
    options_menu.add_checkbutton(label="Extract photo time", onvalue=True, offvalue=False, variable=get_photo_datetime)
    angle_0_360 = Tk.BooleanVar()
    angle_0_360.set(False)
    options_menu.add_checkbutton(label="Angle modulo 360 (0->360)", onvalue=True, offvalue=False, variable=angle_0_360)
    #check_black = Tk.BooleanVar()
    #check_black.set(False)
    #options_menu.add_checkbutton(label="Photos noires (<5Mo)", onvalue=True, offvalue=False, variable=check_black)
    options_menu.add_command(label=u"Test detection", command=test_detection)
    options_menu.add_command(label=u"Calibration (pix/cm)", command=pixel_calibration)
    #TODO: Pour trier ou non les photos
    #sort_photo_num = Tk.BooleanVar()
    #sort_photo_num.set(True)
    #options_menu.add_checkbutton(label="Sort pictures", onvalue=True, offvalue=False, variable=sort_photo_num)

    menubar.add_cascade(label='Options', menu=options_menu)

    #Display the menu
    root.config(menu=menubar)
    root.columnconfigure(1, weight=1, minsize=600)
    root.rowconfigure(1, weight=1)
    #Floating menu (pour afficher des infos pour une tige)
    floatmenu = Tk.Menu(root, tearoff=0)
    #floatmenu.add_command(label="Inverser la base", command=_reverse_tige)
    floatmenu.add_command(label="Réglages", command=show_tige_options)
    floatmenu.add_command(label="Série temporelle", command=show_one_tige)
    floatmenu.add_command(label="Obtenir B", command=show_B)
    floatmenu.add_command(label="Suprimer la base", command=remove_tige)
    floatmenu.add_command(label="Marqueur de fin", command=add_end_marker)
    floatmenu.add_command(label="Supression marqueur de fin", command=remove_end_point)
    floatmenuisopen = False


    def popup(tige_id):
        global floatmenuisopen, cur_tige
         # display the popup menu
        cur_tige = tige_id

        try:
            floatmenu.tk_popup(int(root.winfo_pointerx()), int(root.winfo_pointery()))
            floatmenuisopen = True
        finally:
            # make sure to release the grab (Tk 8.0a1 only)
            floatmenu.grab_release()
            #pass


    #BOTTOM MENU BAR
    buttonFrame = Frame(master=root)
    #buttonFrame.pack(side=Tk.BOTTOM)
    buttonFrame.grid(row=0, column=0, sticky=Tk.W)
    button_traiter = Button(master=buttonFrame, text='Traiter', command=launch_process, state=Tk.DISABLED)
    button_listimages = Button(master=buttonFrame, text="Liste d'images", command=show_image_list, state=Tk.DISABLED)
    button_addtige = Button(master=buttonFrame, text='Ajouter une base', command=_addtige, state=Tk.DISABLED)
    button_supr_all_tige = Button(master=buttonFrame, text='Supprimer les bases', command=_supr_all_tiges, state=Tk.DISABLED)
    button_ouvrir = Button(master=buttonFrame, text='Ouvrir', command=_open_files)
    prog_bar = Progressbar(master=root, mode='determinate')
    #Ajout d'un bouton export to csv
    #button_export = Tk.Button(master=buttonFrame, text=u'Exporter série vers (csv)', command=_export_to_csv, state=Tk.DISABLED)

    """
    button_ouvrir.pack(side=Tk.LEFT)
    button_listimages.pack(side=Tk.LEFT)
    button_addtige.pack(side=Tk.LEFT)
    button_supr_all_tige.pack(side=Tk.LEFT)
    button_traiter.pack(side=Tk.LEFT)
    prog_bar.pack(side=Tk.LEFT, padx=10)
    """
    button_ouvrir.grid(row=0, column=0)
    button_listimages.grid(row=0, column=1)
    button_addtige.grid(row=0, column=2)
    button_supr_all_tige.grid(row=0, column=3)
    button_traiter.grid(row=0, column=4)
    prog_bar.grid(row=2, columnspan=2, sticky=Tk.E+Tk.W)
    
    #button_export.pack(side=Tk.LEFT)
    #figsize=(10,8)
    fig = mpl.figure()
    ax = fig.add_subplot(111)
    
    # a tk.DrawingArea
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.show()
    #canvas.get_tk_widget().pack(side=Tk.BOTTOM, fill=Tk.BOTH, expand=1)
    canvas.get_tk_widget().grid(row=1, columnspan=2, sticky=Tk.W+Tk.E+Tk.N+Tk.S)
    plot_image(cur_image)
    tbar_frame = Tk.Frame(root)
    tbar_frame.grid(row=0, column=1, sticky="ew")
    toolbar = NavigationToolbar2TkAgg( canvas, tbar_frame )
    toolbar.update()
    #canvas._tkcanvas.pack(side=Tk.BOTTOM, fill=Tk.BOTH, expand=1)

    def on_key_event(event):
        print(u'you pressed %s'%event.key)
        key_press_handler(event, canvas, toolbar)

        if event.key == '+':
            global add_tige, nbclick
            add_tige = True
            nbclick = 0

        if event.key == 'right':
            if cur_image + 1 < len(files_to_process):
                plot_image(cur_image + 1, keep_zoom=True)

        if event.key == 'left':
            if cur_image > 1:
                plot_image(cur_image - 1, keep_zoom=True)

        if event.key == 'echap':
            #Cancel add_tige
            if add_tige:
                add_tige = False
                if nbclick == 1:
                    nbclick = 0
                    base_tiges.pop(-1)

                plot_image(cur_image)
                change_button_state()

    def onClick(event):
        global base_tiges, nbclick, add_tige, btige_plt, btige_text
        global floatmenuisopen, tige_id_mapper
        global plt_measure, add_dist_draw, dist_measure_pts, pixel_distance
        #print event

        #Restore focus on the current canvas
        canvas.get_tk_widget().focus_force()

        if event.button == 1:
            #Manage how to add a tige
            if add_tige:
                xy = (event.xdata,event.ydata)
                if xy[0] != None and xy[1] != None:
                    if nbclick == 0:
                        base_tiges.append([xy])
                        nbclick += 1
                        ax.plot(xy[0],xy[1],'r+')
                    else:
                        base_tiges[-1].append(xy)
                        nbclick = 0
                        add_tige = False

                        #Pour les plot append a None to btige
                        btige_plt += [None]
                        btige_text += [None]

                        #Utilisation du tige_id_mapper
                        if len(tige_id_mapper) > 0:
                            tige_id_mapper[max(tige_id_mapper.keys())+1] = len(base_tiges)
                        else:
                            tige_id_mapper[0] = 1

                    plot_basetiges(force_redraw=True)
                    change_button_state()

            if add_dist_draw:
                if dist_measure_pts == []:
                    plt_measure, = ax.plot(event.xdata,event.ydata,'yo-', label='measure', zorder=10)
                    dist_measure_pts += [event.xdata,event.ydata]
                    canvas.draw_idle()
                else:
                    plt_measure.set_data( (dist_measure_pts[0], event.xdata), (dist_measure_pts[1],event.ydata) )
                    canvas.draw_idle()
                    if pixel_distance != None:
                        tmpd = sqrt( (dist_measure_pts[0]-event.xdata)**2 + (dist_measure_pts[1]-event.ydata)**2 )
                        pixel_distance.delete(0, Tk.END)
                        pixel_distance.insert(0, str('%0.2f'%tmpd))

                    dist_measure_pts = []
                    add_dist_draw = False

            #Close floatmenu
            if floatmenuisopen:
                floatmenu.unpost()
                floatmenuisopen = False

        if event.button == 3:
            #Cancel add_tige
            if add_tige:
                add_tige = False
                if nbclick == 1:
                    nbclick = 0
                    base_tiges.pop(-1)

                plot_image(cur_image)
                change_button_state()


    def onPick(event):

        if isinstance(event.artist, mpl.Line2D):
            #print event.mouseevent
            thisline = event.artist
            #xdata = thisline.get_xdata()
            #ydata = thisline.get_ydata()
            #ind = event.ind
            tige_id = thisline.get_label()
            #print('onpick1 line:', zip(take(xdata, ind), take(ydata, ind)))
            try:
                print(u'Selection de la tige %s'%tige_id_mapper(tige_id))
            except:
                print(u'Selection de la tige %i'%(int(tige_id)+1))

            if event.mouseevent.button == 1:
                show_one_tige(tige_id=int(tige_id))

            if event.mouseevent.button == 3:
                popup(int(tige_id))


    cidkey = canvas.mpl_connect('key_press_event', on_key_event)
    canvas.mpl_connect('button_press_event', onClick)
    canvas.mpl_connect('pick_event', onPick)

    def onclose():
        root.destroy()
        mpl.close('all')
        #sys.exit(0)

    root.protocol('WM_DELETE_WINDOW', onclose)


    root.mainloop()

