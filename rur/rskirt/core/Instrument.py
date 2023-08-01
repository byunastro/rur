import numpy as np
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Instrument System
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

def Inst_System(
                wfile,
               
                def_wv_grid,
               includeGALEX,
               includeSDSS,
               include2MASS,
               includeWISE,
               includeHERSCHEL,

               inst_dist_unit,
               inst_dist,
                       
               inc_default,
               azm_default,
               roll_default,
    
               inc_min,
               inc_max,
               inc_del,
               
               azm_min,
               azm_max,
               azm_del,
               
               recordComponents,
               recordPolarization,
               recordStatistics,
               numScatteringLevels,
               
               #@@@!!! fullInstrument
               fov_unit,
               fov_X,
               fov_Y,
               
               pscale_X, 
               pscale_Y,
               
               centre_X,
               centre_Y,

               #@@@!!! SEDInstrument
               save_1d_sed,
               aperture_unit, 
               aperture_min, 
               aperture_max,
               aper_del,
               
               inst_sed_grid_type,
               inst_sed_grid_unit,
               inst_sed_min_wv,
               inst_sed_max_wv,
               inst_sed_N_wv,
    
                indent,
                indent_base
                ):


    N_idt = indent_base
    
    print((N_idt)*indent+'<instrumentSystem type="InstrumentSystem">',file=wfile)
    print((N_idt+1)*indent+'<InstrumentSystem>',file=wfile)
    print((N_idt+2)*indent+'<defaultWavelengthGrid type="WavelengthGrid">',file=wfile)
    
    if def_wv_grid == 'pre-defined':
        print((N_idt+3)*indent+'<PredefinedBandWavelengthGrid includeGALEX="%s" includeSDSS="%s" include2MASS="%s" includeWISE="%s" includeHERSCHEL="%s"/>'%(includeGALEX,includeSDSS,include2MASS,includeWISE,includeHERSCHEL),file=wfile)
        
    print((N_idt+2)*indent+'</defaultWavelengthGrid>',file=wfile)   
    print((N_idt+2)*indent+'<instruments type="Instrument">',file=wfile)   
    
    #!!!
    
    if inc_min != inc_max:
        inc_space = np.linspace(inc_min,inc_max,int((inc_max-inc_min)/inc_del+1)).astype(np.int)
    else:
        inc_space = np.array([inc_min]).astype(np.int)
    
    if azm_min != azm_max:
        azm_space = np.linspace(azm_min,azm_max,int((azm_max-azm_min)/azm_del+1)).astype(np.int)
    else:
        azm_space = np.array([azm_min]).astype(np.int)
        
    if save_1d_sed == True:
        if aperture_min != aperture_max:
            aptr_space = np.linspace(aperture_min,aperture_max,int((aperture_max-aperture_min)/aper_del+1)).astype(np.int)
        else:
            aptr_space = np.array([aperture_min]).astype(np.int)
        
    #!!!
    
    Npix_X = int(fov_X/pscale_X)
    Npix_Y = int(fov_Y/pscale_Y)
    
    i = 0
    while i < len(inc_space):
        inc_i = inc_space[i]
        j = 0
        while j < len(azm_space):
            azm_j = azm_space[j]
            
            if inc_i//10 == 0 and azm_j//10 == 0:
                fname_ij = 'rot_0%s_0%s'%(inc_i,azm_j)
            elif inc_i//10 == 0 and azm_j//10 > 0:
                fname_ij = 'rot_0%s_%s'%(inc_i,azm_j)
            elif inc_i//10 > 0 and azm_j//10 == 0:
                fname_ij = 'rot_%s_0%s'%(inc_i,azm_j)  
            else:
                fname_ij = 'rot_%s_%s'%(inc_i,azm_j)  
            
            print((N_idt+3)*indent+'<FullInstrument instrumentName="%s" distance="%s %s" inclination="%s deg" azimuth="%s deg" roll="%s deg" fieldOfViewX="%s %s" numPixelsX="%s" centerX="%s %s" fieldOfViewY="%s %s" numPixelsY="%s" centerY="%s %s" recordComponents="%s" numScatteringLevels="%s" recordPolarization="%s" recordStatistics="%s"/>'%(fname_ij,inst_dist,inst_dist_unit,inc_i,azm_j,roll_default,fov_X,fov_unit,Npix_X,centre_X,fov_unit,fov_Y,fov_unit,Npix_Y,centre_Y,fov_unit,recordComponents,numScatteringLevels,recordPolarization,recordStatistics),file=wfile)
            
            if save_1d_sed == True:
                k = 0
                while k < len(aptr_space):
                    aptr_k = aptr_space[k]
                    if aptr_k//10 == 0:
                        fname_ijk = fname_ij + '_0%s'%(aptr_k)
                    else:
                        fname_ijk = fname_ij + '_%s'%(aptr_k)
                
                    print((N_idt+3)*indent+'<SEDInstrument instrumentName="%s" distance="%s %s" inclination="%s deg" azimuth="%s deg" roll="%s deg" radius="%s %s" recordComponents="%s" numScatteringLevels="%s" recordPolarization="%s" recordStatistics="%s">'%(fname_ijk,inst_dist,inst_dist_unit,inc_i,azm_j,roll_default,aptr_k,aperture_unit,recordComponents,numScatteringLevels,recordPolarization,recordStatistics),file=wfile)
                    print((N_idt+4)*indent+'<wavelengthGrid type="WavelengthGrid">',file=wfile)
                    
                    if inst_sed_grid_type == 'lin':
                        print((N_idt+5)*indent+'<LinWavelengthGrid minWavelength="%s %s" maxWavelength="%s %s" numWavelengths="%s"/>'%(inst_sed_min_wv,inst_sed_grid_unit,inst_sed_max_wv,inst_sed_grid_unit,inst_sed_N_wv),file=wfile)
                    elif inst_sed_grid_type == 'log':
                        print((N_idt+5)*indent+'<LogWavelengthGrid minWavelength="%s %s" maxWavelength="%s %s" numWavelengths="%s"/>'%(inst_sed_min_wv,inst_sed_grid_unit,inst_sed_max_wv,inst_sed_grid_unit,inst_sed_N_wv),file=wfile)
                        
                    print((N_idt+4)*indent+'</wavelengthGrid>',file=wfile)
                    print((N_idt+3)*indent+'</SEDInstrument>',file=wfile)
                    
                    k += 1
                    
            j += 1
        i += 1
        
    print((N_idt+2)*indent+'</instruments>',file=wfile)  
    print((N_idt+1)*indent+'</InstrumentSystem>',file=wfile)     
    print((N_idt)*indent+'</instrumentSystem>',file=wfile)           
            
    return N_idt










