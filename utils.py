#Import necesary modules
#Use shorter names (np, pd, plt) instead of full (numpy, pandas, matplotlib.pylot) for convenience
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
import h5py
import numpy as np
import pdb
import numpy.ma as ma
from astropy.time import Time
import pyproj
import xarray as xr
from scipy.interpolate import griddata

def get_atl10_freeboards(fileT, beam=beamStr, minFreeboard=0, maxFreeboard=10, epsg_string="3411"):
    """ Pandas/numpy ATL10 reader
    Original function written by Alek Petty, June 2018 (alek.a.petty@nasa.gov)

    I've picked out the variables from ATL10 I think are of most interest to sea ice users, 
    but by no means is this an exhastive list. 
    See the xarray or dictionary readers to load in the more complete ATL10 dataset
    or explore the hdf5 files themselves (I like using the app Panpoly for this) to see what else you might want
    
    Args:
        fileT (str): File path of the ATL10 dataset
        beamStr (str): ICESat-2 beam (the number is the pair, r=strong, l=weak)
        maxFreeboard (float): maximum freeboard (meters)
    
    Optional args:
        epsg_string (str): EPSG string for projecting data (default of 3411 north polar stereo)

    returns:
        pandas dataframe
        
    Versions:
        v1: June 2018
        v2: June 2020 - cleaned things up, changed the time function slightly to be consistent with Ellen's ATL07 reader.

    """

    print('ATL10 file:', fileT)
    
    f1 = h5py.File(fileT, 'r')
    
    # Freeboards
    freeboard=f1[beam]['freeboard_beam_segment']['beam_freeboard']['beam_fb_height'][:]
    ssh_flag=f1[beam]['freeboard_beam_segment']['height_segments']['height_segment_ssh_flag'][:]
    
    # Freeboard confidence and freeboard quality flag
    freeboard_confidence=f1[beam]['freeboard_beam_segment']['beam_freeboard']['beam_fb_confidence'][:]
    freeboard_quality=f1[beam]['freeboard_beam_segment']['beam_freeboard']['beam_fb_quality_flag'][:]
    freeboard_sigma=f1[beam]['freeboard_beam_segment']['beam_freeboard']['beam_fb_sigma'][:]
    
    # Getting a lot of high values/NaNs - why is that?!
    freeboard_sigma[np.where(freeboard_sigma>0.2)]=np.nan
    # Getting a lot of NaNs - why is that?!
    # As a temporary measure set the value to the max sigma in the granule.

    freeboard_sigma[np.isnan(freeboard_sigma)]=np.nanmax(freeboard_sigma)

    
    # Along track distance from the equator (convert to kilometers)
    seg_x = f1[beam]['freeboard_beam_segment']['beam_freeboard']['seg_dist_x'][:]*0.001
    
    # Height segment ID (10 km segments)
    height_segment_id=f1[beam]['freeboard_beam_segment']['beam_freeboard']['height_segment_id'][:]
    
    lons=f1[beam]['freeboard_beam_segment']['beam_freeboard']['longitude'][:]
    lats=f1[beam]['freeboard_beam_segment']['beam_freeboard']['latitude'][:]
    
    crs = pyproj.CRS.from_string("epsg:"+epsg_string)
    mapProj=pyproj.Proj(crs)

    xpts, ypts=mapProj(lons, lats)

    # Time since the start of the granule
    #deltaTimeRel=delta_time-delta_time[0]
    
    # Delta time in gps seconds
    delta_time = f1[beam]['freeboard_beam_segment']['beam_freeboard']['delta_time'][:]
    
    # #Add this value to delta time parameters to compute full gps_seconds
    atlas_epoch=f1['/ancillary_data/atlas_sdp_gps_epoch'][0] 
    gps_seconds = atlas_epoch + delta_time

    ## Use astropy to convert GPS time to UTC time
    tiso=Time(gps_seconds,format='gps').utc.datetime

    dF = pd.DataFrame({'freeboard':freeboard, 'freeboard_sigma':freeboard_sigma, 'freeboard_quality':freeboard_quality, 'ssh_flag':ssh_flag, 'lon':lons, 'lat':lats,
                     'height_segment_id':height_segment_id,'datetime': tiso, 'seg_x':seg_x, 'xpts':xpts, 'ypts':ypts})
    
    dF = dF[(dF['freeboard']>=minFreeboard)]
    dF = dF[(dF['freeboard']<=maxFreeboard)]

    # Could add extra filters based on confidence and/or quality flag?
    
    # Reset row indexing
    dF=dF.reset_index(drop=True)

    return dF

def convert_to_thickness(dF, snowDepthVar='snowDepth', snowDensityVar='snowDensity', outVar='iceThickness', rhoi=1):
    """ Convert freeboard to thickness
    
    Args:
        dF (dataframe): dataframe containing the data

    Optional args:
        snowDepthVar (str): snow depth variable of choosing
        snowDensityVar (str): snow density variable of choosing
        outVar (str): output string
        rhoi (int): ice density option

    """
    
    # Need to copy arrays or I think it will overwrite the pandas column!
    iceDensityT=np.ones_like(dF['freeboard'].values)*916.
    
    if 'ice_density_1' not in dF:
        dF['ice_density_1'] = pd.Series(iceDensityT, index=dF.index)

    if (rhoi==2):
        # set lower density for MYI
        try:
            iceType=np.copy(dF['ice_type'].values)
        except:
            print('No ice type information')
        iceDensityT[np.where(iceType>0.5)]=882.
        if 'ice_density_2' not in dF:
            dF['ice_density_2'] = pd.Series(iceDensityT, index=dF.index)

    elif (rhoi==3):
        # set lower density for MYI
        try:
            iceType=np.copy(dF['ice_type'].values)
        except:
            print('No ice type information')
        iceDensityT[np.where(iceType>0.5)]=899.
        if 'ice_density_3' not in dF:
            dF['ice_density_3'] = pd.Series(iceDensityT, index=dF.index)

    ice_thickness = freeboard_to_thickness(np.copy(dF['freeboard'].values), np.copy(dF[snowDepthVar].values), np.copy(dF[snowDensityVar].values), iceDensityT)

    # add to dataframe
    dF[outVar] = pd.Series(np.array(ice_thickness), index=dF.index)
   
    return dF



def freeboard_to_thickness(freeboardT, snow_depthT, snow_densityT, rho_i=925., rho_w=1024.):
    """
    Hydrostatic equilibrium equation to calculate sea ice thickness 
    from freeboard, snow depth/density, sea water and ice density

    Args:
        freeboardT (var): ice freeboard
        snow_depthT (var): snow depth
        snow_densityT (var): final snow density
    Optional args
        rho_i (var): sea ice density estimate
        rho_w (var): sea water density estimate

    Returns:
        ice_thicknessT (var): ice thickness dereived using hydrostatic equilibrium

    """

    # set snow depth to freeboard where snow depth greater than freeboard.
    snow_depthT[snow_depthT>freeboardT]=freeboardT[snow_depthT>freeboardT]

    ice_thicknessT = (rho_w/(rho_w-rho_i))*freeboardT - ((rho_w-snow_densityT)/(rho_w-rho_i))*snow_depthT

    return ice_thicknessT

def calc_thickness_uncertainty(dF, snowDepthVar, snowDensityVar, iceDensityVar, outVar, ice_density_assumptions=[], snow_depth_assumptions=[],
    snow_density_assumptions=[], snow_depth_unc_sys_default=0.1, snow_density_unc_sys_default=30, ice_density_unc_sys_default=10, freeboard_unc_sys_default=0.01,
    seawater_density=1024., snow_depth_unc_random=0.05, snow_density_unc_random=40, 
    ice_density_unc_random=10., water_density_unc_random=0.5):
    """ Calculate sea ice thickness uncertainty using simple error propogation. 

        The approach and justificaiton is described more in Petty, A. A., N. T. Kurtz, R. Kwok, T. Markus, T. A. Neumann (2020), 
        Winter Arctic sea ice thickness from ICESat‐2 freeboards, Journal of Geophysical Research: Oceans, 125, e2019JC015764. doi:10.1029/2019JC015764 


        Args:
            dF (dataframe): dataframe containing the data
            snowDepthVar (str): snow depth variable of choosing
            snowDensityVar (str): snow density variable of choosing
            outVar (str): output string
            rhoi (int): ice density option
        Optional args:
            snow_depth_assumptions (list): list of snow depth assumptions included as columns in the dataframe
            snow_density_assumptions (list): list of snow density assumptions included as columns in the dataframe
            ice_density_assumptions (list): list of ice density assumptions included as columns in the dataframe
            snow_depth_unc_sys_default (float): default assumption of systematic snow depth uncertianity in the absense of input assumptions (m)
            snow_density_unc_sys_default (float): default assumption of systematic snow density uncertianity in the absense of input assumptions (m)
            ice_density_unc_sys_default (float): default assumption of systematic ice density uncertianity in the absense of input assumptions (kg/m3)
            freeboard_unc_sys_default (float): default assumption of systematic freeboard uncertianity in the absense of input assumptions (m)
            seawater_density (float): sea water density (kg/m3)
            snow_depth_unc_random (float): sea depth random uncertainty (m)
            snow_density_unc_random (float): snow density random uncertainty (kg/m3)
            ice_density_unc_random (float): ice density random uncertainty (kg/m3)
            water_density_unc_random (float): sea water density random uncertainty (kg/m3)

        Versions:
            (03/20/2021)- 

    """

    # Need to copy arrays or it will overwrite the pandas column!
    freeboard=np.copy(dF['freeboard'].values)    
    snow_depth=np.copy(dF[snowDepthVar].values)
    snow_density=np.copy(dF[snowDensityVar].values)
    ice_density=np.copy(dF[iceDensityVar].values)

    #------- Random uncertainty calculation

    # freeboard
    # This is the ice height precision and the combined lead uncertainties
    freeboard_unc_random=np.copy(dF['freeboard_sigma'].values)
    
    freeboard_thickness_unc_random = (freeboard_unc_random**2)* \
        (seawater_density/(seawater_density-ice_density))**2
    
    # snow depth
    snow_depth_thickness_unc_random = (snow_depth_unc_random**2)* \
        ((snow_density-seawater_density)/(seawater_density-ice_density))**2
    
    # snow density
    snow_density_thickness_unc_random = (snow_density_unc_random**2)* \
        (snow_depth/(seawater_density-ice_density))**2
    
    # sea water density (later neglected)
    water_density_thickness_unc_random = (water_density_unc_random**2)* \
        (((freeboard-snow_depth)/(seawater_density-ice_density))+
                                    (((snow_depth*seawater_density)-(freeboard*seawater_density)-(snow_depth*snow_density))/(seawater_density-ice_density)**2))**2
    
    # sea ice density
    ice_density_thickness_unc_random = (ice_density_unc_random**2)* \
        (((freeboard*seawater_density)+(snow_depth*snow_density)-(snow_depth*seawater_density))/(seawater_density-ice_density)**2)**2

    # Combine (dropped water density uncertainity as negligible)
    random_uncertainty_squared = (freeboard_thickness_unc_random+ \
                            snow_depth_thickness_unc_random + \
                            snow_density_thickness_unc_random + \
                            ice_density_thickness_unc_random)


    #------- Systematic uncertainty calculation

    # Snow depth
    if len(snow_depth_assumptions)>1:
        # If there are a series of input assumptions available
        snow_depth_unc_sys=dF[snow_depth_assumptions].std(axis=1)
    else:
        #Just prescribe/guess the systematic snow depth unceratinity
        snow_depth_unc_sys = snow_depth_unc_sys_default
    
    #print('snow depth unc', snow_depth_unc_sys)
    snow_depth_thickness_unc_sys = (snow_depth_unc_sys**2)* \
        ((snow_density-seawater_density)/(seawater_density-ice_density))**2

    # Snow density
    if len(snow_density_assumptions)>1:
        # If there are a series of input assumptions available
         snow_density_unc_sys=dF[snow_density_assumptions].std(axis=1)
    else:
        #Just prescribe/guess the systematic snow depth unceratinity
        snow_density_unc_sys = snow_density_unc_sys_default

    snow_density_thickness_unc_sys = (snow_density_unc_sys**2)* \
        (snow_depth/(seawater_density-ice_density))**2

    # Sea ice density
    if len(ice_density_assumptions)>1:
        # If there are a series of input assumptions available
         ice_density_unc_sys=dF[ice_density_assumptions].std(axis=1)
    else:
        #Just prescribe/guess the systematic snow depth unceratinity
        ice_density_unc_sys = ice_density_unc_sys_default
   
    ice_density_thickness_unc_sys = (ice_density_unc_sys**2)* \
        (((freeboard*seawater_density)+(snow_depth*snow_density)-(snow_depth*seawater_density))/(seawater_density-ice_density)**2)**2

    # Sea ice freeboard
    # Include a systematic freeboard uncertaiity, expected from ice height or sea level determination biases
    freeboard_unc_sys=freeboard_unc_sys_default

    freeboard_thickness_unc_sys = (freeboard_unc_sys**2)* \
        (seawater_density/(seawater_density-ice_density))**2

    # Combine systmatic uncertainities
    sys_uncertainty_squared = (snow_depth_thickness_unc_sys + \
                            snow_density_thickness_unc_sys + \
                            ice_density_thickness_unc_sys + 
                            freeboard_thickness_unc_sys)
    
    # Add uncertainties to dataframe
    dF['freeboard_unc'] = pd.Series(np.sqrt((freeboard_unc_random)**2+(freeboard_unc_sys)**2), index=dF.index)
    dF[outVar+'random'] = pd.Series(np.sqrt(random_uncertainty_squared), index=dF.index)
    dF[outVar+'sys'] = pd.Series(np.sqrt(sys_uncertainty_squared), index=dF.index)
    dF[outVar] = pd.Series(np.sqrt(random_uncertainty_squared+sys_uncertainty_squared), index=dF.index)
   
    return dF


def grid_NESOSIM_to_freeboard(dF, dNday, epsg_string='3411', outSnowVar='snowDepthN', outDensityVar='snowDensityN', returnMap=False):
    """
    Assign relevant NESOSIM snow data file and assign to freeboard values

    Args:
        dF (data frame): Pandas dataframe
        dNday (data frame): NESOSIM data for the given day

    Optional args:
        outSnowVar (string): Name of snow depth column
        outDensityVar (string): Name of snow density column
        epsg_string (str): EPSG string for projecting data (default of 3411 north polar stereo)
        returnMap (boolean): if true return x/y coordinates
        
    Returns:
        dF (data frame): dataframe updated to include colocated NESOSIM (and dsitributed) snow data

    To do:


    """
    print(dNday)
    lonsN = np.array(dNday.longitude)
    latsN = np.array(dNday.latitude)

    crs = pyproj.CRS.from_string("epsg:"+epsg_string)
    mapProj=pyproj.Proj(crs)

    xptsN, yptsN=mapProj(lonsN, latsN)

    try:
        # New (v1.1) variable names
        snowDepthNDay = np.array(dNday.snow_depth)
    except:
        # old way
        snowDepthNDay = np.array(dNday.snowDepth)

    try:
        snowDensityNDay= np.array(dNday.snow_density)
    except:
        snowDensityNDay= np.array(dNday.density)

    try:
        iceConcNDay = np.array(dNday.ice_concentration)
    except:
        iceConcNDay = np.array(dNday.iceConc)
    
    # Remove data where snow depths less than 0 (masked).
    mask=np.where((snowDepthNDay>0.01)&(snowDepthNDay<1)&(iceConcNDay>0.01)&np.isfinite(snowDensityNDay))

    snowDepthNDay = snowDepthNDay[mask]
    snowDensityNDay = snowDensityNDay[mask]
    xptsN = xptsN[mask]
    yptsN = yptsN[mask]

    # Use projected coordinates to find nearest NESOSIM data
    snowDepthGISs = griddata((xptsN, yptsN), snowDepthNDay, (dF['xpts'].values, dF['ypts'].values), method='nearest') 
    snowDensityGISs = griddata((xptsN, yptsN), snowDensityNDay, (dF['xpts'].values, dF['xpts'].values), method='nearest')
   
    # Apply to dataframe
    dF[outSnowVar] = pd.Series(snowDepthGISs, index=dF.index)
    dF[outDensityVar] = pd.Series(snowDensityGISs, index=dF.index)

    
    if (returnMap==1):
        return dF, xptsN, yptsN, dNday, 
    else:
        return dF

def get_nesosim(dateStr, snowPathT):
    """ 
    Grab the NESOSIM data for the relevant date

    NB: rarely does an ICESat-2 granule span more than two day but we ignore it if it does and just use the first day

    """
    
    print(snowPathT)

    # Open file
    dN = xr.open_dataset(snowPathT)
    
    # Get NESOSIM snow depth and density data for that date
    dNday = dN.sel(day=int(dateStr))
    
    return dNday
