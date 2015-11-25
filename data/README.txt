Data for the following manuscript:

William J M Probert, Katriona Shea, C J Fonnesbeck, M C Runge, T E Carpenter, S Dürr, M. G Garner, N Harvey, M A. Stevenson, C T. Webb, M Werkman, M J Tildesley, M J Ferrari (in prep) Decision-making for foot-and-mouth disease control: objectives matter.  


Description
---------------

These data are from simulations of a foot-and-mouth disease (FMD) outbreak
using five independently developed disease spread models (listed below).  Each
model ran 100 simulations each (note 1) for five different control types that
are listed below (note 2).  These data are used to illustrate the dependence of
a recommended control action on the choice of management objective.  

Demographic parameters for the hypothetical outbreak scenario were chosen to be
consistent with the county of Cumbria in the UK.  Such parameters included the
sizes of the farms and the proportion of sheep and cattle in each farm (other
cloven-hooved species were ignored in this analysis).  Spread was simulated
across 7837 farms with a spatial distribution consistent with Cumbria, UK.  All
models were run from the start of the control program with 10 infected farms in
the population at this time, consistent with a single point introduction and 10
day interval until FMD is notified.  After the first farm was reported with
infection it was assumed a livestock movement ban was implemented between
90-100% efficacy.  Culling of infected premises was performed under all control
actions.  Only one of these management actions was implemented per simulated
outbreak.  Culling was constrained to a maximum of 50 farms per day and
vaccination had a capacity of 10,000 animals per day.  Vaccine efficacy was
between 80-90%, specific to each model.  For each simulation, the duration of
the simulated outbreak, the total number of livestock culled by species, and
the total number of vaccine doses administered were recorded.  



File contents
---------------

Each file contains the following columns:

run: integer
    The simulation run for a particular model and simulated control action.  


model: string

    The model used to run the simulation.  These are capital letters A-E
    denoting a particular model.  Models have been anonymised, yet the five
    models used to run these analyses are (in alphabetical order): 
    
    1) AusSpread, developed by the Australian Government Department of
    Agriculture, Forestry, and Fisheries (Garner and Beckett, 2005; 
    Roche et al. 2014)
    
    2) The Davis Animal Disease Spread model developed at the University of
    California, Davis (DADS; Bates et al, 2003)
    
    3) Interspread Plus, developed at Massey University, New Zealand (ISP;  
    Sanson, 1993; Stern, 2003; Stevenson et al, 2013); 
    
    4) The North American Animal Disease Spread Model jointly developed by the
    US and Canada, and with continued development by the Animal and Plant
    Health Inspection Service of the United States Department of Agriculture
    (NAADSM; Harvey et al, 2007); 
    
    5) The Warwick model, originally developed at Cambridge University during
    the 2001 UK outbreak but then further developed at Warwick University from
    2003 onward (Keeling et al, 2001; Tildesley et al, 2008).  


control: string

    The control action used for the whole course of the simulated outbreak  
    (in addition to a movement ban being simulated).  The five control actions
    are
    ip: culling of infected farms only.  
    
    dc: culling of infected farms and of those that have been identified as at
    risk through tracing of dangerous contacts.  
    
    rc: culling of infected farms and of all those within 3km of each infected
    farm
    
    v03: culling of all infected farms and vaccination of all farms within 3km
    of each infected farm
    
    v10: culling of all infected farms and vaccination of all farms within 10km
    of each infected farm.  


duration: integer
    
    The duration of the simulated outbreak (in days) from the first reported
    case to when there were no more infected or exposed animals in the 
    simulation.  


cattle_culled: integer
    
    The total number of cattle culled in all farms at the end of the outbreak.  


sheep_culled: integer
    
    The total number of sheep culled in all farms at the end of the outbreak.  


cattlesheep_culled: integer
    
    The total number of animals culled on mixed sheep and cattle farms (if the
    model allowed for this).  Only one model reported this specific mix of
    farms separately to the number of culled cattle or sheep.  


cattle_vacc: integer
    
    The total number of cattle vaccinated in all farms at the end of the
    outbreak.  



Notes
---------------

1) Model A has only 99 simulation runs for each simulated control action.  
2) For model B it was not possible to simulate ring culling, this control action is omitted in this model.  



References
---------------


Bates, T. W., M. C. Thurmond, and T. E. Carpenter (2003). Description of an epidemic simulation model for use in evaluating strategies to control an outbreak of foot-and-mouth disease. American Journal of Veterinary Research 64(2), 195–204.

Garner, M. G. and S. D. Beckett (2005). Modelling the spread of foot-and-mouth disease in australia. Australian Veterinary Journal 83(12), 758–766.

Harvey, N., A. Reeves, M. A. Schoenbaum, F. J. Zagmutt-Vergara, C. Dubé, A. E. Hill, B. A. Corso, W. B. McNab, C. I. Cartwright and M. D. Salman (2007) The North American Animal Disease Spread Model: A simulation model to assist decision making in evaluating animal disease incursions.  Preventive Veterinary Medicine (82) 176–197.  

Keeling, M. J., M. E. J. Woolhouse, D. J. Shaw, L. Matthews, M. Chase-Topping, D. T. Haydon, S. J. Cornell, J. Kappey, J. Wilesmith, and B. T. Grenfell (2001). Dynamics of the 2001 UK foot and mouth epidemic: Stochastic dispersal in a heterogeneous landscape. Science 294, 813–817.

Roche, S. E., M. G. Garner, R. M. Wicks, I. J. East, K. de Witte (2014) How to resources influence control measures during a simulated outbreak of foot and mouth disease in Australia?  Preventive Veterinary Medicine. 113. 436-446.  

Sanson, R. L. (1993) The development of a decision support system for an animal disease emergency. PhD thesis, Massey University, Palmerston North, NZ.

Stern, M., (2003) InterSpread Plus User Guide. Institute of Veterinary, Animal, and Biomedical Sciences, Massey University, Palmerston North, New Zealand.  

Stevenson M. A., Sanson R. L., Stern M. W., O’Leary B. D., Sujau M., Moles-Benfell N., and Morris R. S. (2013) InterSpread Plus: A spatial and stochastic simulation model of disease in animal populations. Preventive Veterinary Medicine. 109. 10–24.

Tildesley, M. J.,  R. Deardon, N. J. Savill, P. R. Bessell, S. P. Brooks, M. E. J. Woolhouse, B. T. Grenfell, M. J. Keeling (2008) Accuracy of models for the 2001 foot-and-mouth epidemic.  Proceedings of the Royal Society of London B: Biological Sciences. 275 (1641) 1459–1468.



Contributions
---------------

T E Carpenter, S Dürr, M. G Garner, N Harvey, M A. Stevenson, M Werkman, and M J Tildesley generated the simulation data.  M J Ferrari and W J M Probert processed the simulation data to put it in a standard format.  



Contact
---------------

Questions or comments may be forwarded to:

William J M Probert
The Center for Infectious Disease Dynamics
The Pennsylvania State University
University Park, PA, 16802, USA

phone:  +1 814 441 3204
web:    www.probert.co.nz
email:  wjp11@psu.edu

---------------
