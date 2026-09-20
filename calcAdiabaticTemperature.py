import thermosteam as tmo
import biosteam as bst
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable



####################################################################
### get feedstock data
df = pd.read_csv("Outputs/biomassFeedstockData.csv")

### constants
# MW, g / mol
MW_C, MW_H, MW_O, MW_N, MW_S = 12.011, 1.008, 16.0, 14.0, 32.06


####################################################################
### FUNCTIONS
####################################################################
### do function to get Hf from HHV
def delta_Hf_from_HHV(c, h, o, n, s, HHV_Jmol):
    """
    Calculate standard enthalpy of formation from HHV for biomass.
    Complete combustion: CxHyOzNaSb(s) + O2 -> CO2 + H2O(l) + NO2 + SO2
    # corrected for ASTM D 5865-10a, which includes an explicit Acid Correction step
    Complete combustion: CxHyOzNaSb(s) + O2 -> CO2 + H2O(l) + N2 + SO2

    """    
    # Tabulated standard enthalpies of formation (J/mol)
    dHf_CO2 = -393510  # J/mol
    dHf_H2O_l = -285830  # J/mol for LIQUID water
    dHf_H2O_g = -241820  # J/mol for GASEOUS water
    dHf_NO2 = 33100  # J/mol
    dHf_SO2 = -296800  # J/mol
    # correction for calorific analysis using ASTM D 5865-10a:
    dHf_N2 = 0  # J/mol


    # Products in complete combustion
    n_CO2 = c
    n_H2O = h / 2
    n_NO2 = n  # N -> NO2
    n_N2  = n / 2 # N --> N2
    n_SO2 = s  # S -> SO2
    
    # Hess's law
    delta_Hc_Jmol = -HHV_Jmol  # HHV is negative of heat of combustion
    sum_products = (n_CO2 * dHf_CO2 + n_H2O * dHf_H2O_l + 
                   n_N2 * dHf_N2 + n_SO2 * dHf_SO2)
    delta_Hf_fuel_Jmol = sum_products - delta_Hc_Jmol
    
    return delta_Hf_fuel_Jmol


### BIOMASS adiabatic temperature function
def getFlueGasTemperature(c,h,o,n,s,ash,moisture,HHV,excess,CO_to_COx,NO2_to_N,heat_loss,T_preheat,name):

    ####################################################
    ###### CREATE BIOMASS COMPONENT
    ######
    # After normalization, format coefficients
    c_fmt = round(c, 6)
    h_fmt = round(h, 6)
    o_fmt = round(o, 6)
    n_fmt = round(n, 6)
    s_fmt = round(s, 6)

    # Convert to string with explicit decimal format (no scientific notation)
    formula_str = f'C{c_fmt:.6f}H{h_fmt:.6f}O{o_fmt:.6f}N{n_fmt:.6f}S{s_fmt:.6f}'
    # print
    # print(f"Biomass empirical formula: {formula_str}")

    ### Define chemicals (including custom Biomass) and thermo
    Biomass = bst.Chemical(
        'Biomass',
        search_db=False,
        phase='s',
        formula=formula_str,
        common_name=name
    )

    # flag if biomass MW too large
    if Biomass.MW > 50:  # Normal biomass MW normalized to C=1 should be ~20-30
        print(f"Input formula string: C{c}H{h}O{o}N{n}S{s}")
        print(f"Biomass.formula: {Biomass.formula}")
        print(f"Biomass.atoms: {Biomass.atoms}")
        print(f"Biomass.MW: {Biomass.MW}")

    # HHV (convert MJ/kg --> to J / mol, dry, ash-free)
    Biomass.HHV = HHV * 1e6 / 1e3 * Biomass.MW
    dHf_biomass = delta_Hf_from_HHV(c, h, o, n, s, Biomass.HHV)
    # replace
    Biomass.Hf = dHf_biomass


    ####################################################
    ###### CREATE OTHER INLET COMPONENTS
    ### assign Cp
    # Source: https://doi.org/10.1016/j.fuel.2013.07.086
    # result from paper: The relative difference of heat capacity between the different biomasses was significant but lower than 20%, with values ranging from 1300 to 2000 J kg−1 K−1
    a_JgK = 1400 / 1e3  # J / g / K
    a = a_JgK * Biomass.MW

    Biomass.Cn.add_model(a)  # linear: Cp = a + b*T
    Biomass.Cn.method = "USER_METHOD"
    # Biomass.get_missing_properties()
    # fill missing properties
    Biomass.default()

    # Define ash as an inert chemical
    Ash = bst.Chemical(
        'Ash',
        search_db=False,
        phase='s',
        formula='',  # No formula, or use a representative one like 'SiO2'
        MW=60.08,  # Approximate MW (e.g., SiO2)
        Hf=0,  # No heat of formation contribution
    )
    Ash.Cn.add_model(0.84 * Ash.MW)  # Cp ≈ 840 J/kg/K for typical ash: https://doi.org/10.1016/j.wasman.2025.114752 
    Ash.default()

    ### define remaining chemicals
    H2O = bst.Chemical('H2O')#, phase='g')
    O2  = bst.Chemical('O2',  phase='g')
    N2  = bst.Chemical('N2',  phase='g')
    CO2 = bst.Chemical('CO2', phase='g')
    CO  = bst.Chemical('CO',  phase='g')
    NO2 = bst.Chemical('NO2', phase='g')
    NO  = bst.Chemical('NO',  phase='g')
    SO2 = bst.Chemical('SO2', phase='g')

    # create and compile
    chemicals = tmo.Chemicals([Biomass, H2O, O2, N2, CO2, CO, 
                               NO2, NO, SO2, Ash])
    chemicals.compile()
    bst.settings.set_thermo(chemicals, cache=False)


    ####################################################
    ###### Combustion reaction + adiabatic temperature
    ######

    # create string of reaction
    # percentage of CO to CO+CO2; all carbon goes to these; same for N
    rxn_str = 'Biomass + O2 -> H2O + SO2 + ' \
            + str(1-CO_to_COx) + 'CO2 + ' \
            + str(CO_to_COx) + 'CO + ' \
            + str(n_fmt * NO2_to_N) + 'NO2 + ' \
            + str(n_fmt * (1-NO2_to_N)/2) + 'N2'
            
    reaction = bst.Reaction(
        rxn_str,
        reactant='Biomass',
        X=1.0,
    )
    # adjust stoich to satisfy C,H,O,N
    reaction.correct_atomic_balance(constants=['CO2', 'CO', 'NO2', 'N2']) 

    # set molar flows
    n_biomass = 1
    n_O2 = -reaction.stoichiometry[reaction.chemicals.index('O2')] * (1 + excess) * n_biomass

    # create ash molar flows
    n_ash = n_biomass * ash / (1 - ash) * (Biomass.MW / Ash.MW)

    # Convert moisture from mass% to molar basis
    # If moisture is 5% by mass, then for 100g total:
    # - 5g is water
    # - 95g is dry biomass
    # Moles of water per mole of dry biomass = (mass_water/MW_water) / (mass_biomass/MW_biomass)
    #
    n_dry_biomass = n_biomass + n_ash
    MW_dry = (n_biomass * Biomass.MW + n_ash * Ash.MW) / n_dry_biomass

    if moisture > 0:
        n_H2O_moisture = n_dry_biomass * (moisture / chemicals.H2O.MW) / ((1 - moisture) / MW_dry)
    else:
        n_H2O_moisture = 0


    ####################################################
    ###### THERMODYNAMICS CALCULATIONS BELOW
    ######
    # Step 1a: preheated air (N2 + O2)
    air = bst.Stream(
        '', O2=n_O2, N2=79/21*n_O2,
        T=T_preheat, P=101325, phase='g', units='kmol/hr',
    )

    # Step 1b: biomass + moisture + ash at ambient
    fuel = bst.Stream(
        '', Biomass=n_biomass, H2O=n_H2O_moisture, Ash=n_ash,
        T=298.15, P=101325, phase='g', units='kmol/hr',
    )

    # Step 1c: combine into the single inlet stream used downstream
    s1 = bst.Stream('')
    s1.mix_from([air, fuel])
    # inlet enthalpy (NOT zero, since H2O forced gas)
    H_inlet_gas_basis = s1.H

    # react!
    reaction.adiabatic_reaction(s1)
    H_outlet_gas_basis = s1.H   # sensible H after reaction, same water-as-gas basis

    # Step 2: correct BOTH sides identically for moisture actually starting as liquid
    # T here is 298.15 K since that is the temperature for vaporization
    Hvap_H2O = chemicals.H2O.Hvap(298.15)
    evaporation_correction = n_H2O_moisture * Hvap_H2O

    # inlet didnt have water as gas, so remove that
    H_inlet = H_inlet_gas_basis - evaporation_correction
    # at the outlet, initial moisture was liquid, which absorbed the heat by evaporating
    H_corrected = H_outlet_gas_basis - evaporation_correction   # this is your outlet, on the corrected basis

    # Q_released: heat released by combustion, now correctly isolated
    Q_released = H_corrected - H_inlet
    # the bottom values should be more or less equal
    # print(f"Q_released: {Q_released:.1f} J   vs   HHV: {Biomass.HHV * n_biomass:.1f} J")  # sanity check

    # updates stream by removing heat energy that would've been absorbed by moisture evaporation
    s1.H = H_corrected
    T_gas_only = s1.T

    # Step 3: real flash
    # this step allows the water to choose its VLE phase under these conditions
    # i.e., it doesn't have to be gas (but it could and likely is at high temps)
    s2 = bst.MultiStream('', phases='lgs', T=T_gas_only, P=101325, units='kmol/hr')
    for chem_ID in ['O2', 'N2', 'CO2', 'CO', 'NO2', 'SO2', 'H2O']:
        amt = s1.imol[chem_ID]
        if amt > 0:
            s2.imol['g', chem_ID] = amt
    s2.imol['s', 'Ash'] = s1.imol['Ash']

    # apply VLE update
    s2.vle(P=101325, H=H_corrected)
    H_outlet_flashed = s2.H

    # heat loss: remove heat_loss fraction of the heat energy released
    H_after_loss = H_outlet_flashed - heat_loss * Q_released
    # update VLE calculations
    s2.vle(P=101325, H=H_after_loss)

    # store flue gas temperature (adiabatic, if heat_loss = 0) and outlet stream
    T_adiabatic = s2.T
    outlet = s2

    # do liquid/gas water VLE accounting
    liquid_H2O = s2.imol['l', 'H2O']
    gas_H2O    = s2.imol['g', 'H2O']
    total_H2O  = liquid_H2O + gas_H2O
    # print
    if liquid_H2O > 1e-6:
        print(f"[LIQUID WATER @ equilibrium] {name}: liquid H2O = {liquid_H2O:.4f} kmol/hr " + "\n" +
              f"({100*liquid_H2O/total_H2O:.1f}% of total water), T = {T_adiabatic-273.15:.2f}°C")

    ######

    # some printing
    print_ON = 0
    if print_ON:
        print("After adiabatic combustion:")
        # s1.show()
        print("Adiabatic flame temperature [K]:", T_adiabatic)
        print("")


    return T_adiabatic, reaction, Biomass, outlet


### initialize composition
def initializeComposition(row, includeNS_ON=1, includeMoisture_ON=1, includeAsh_ON=1):
    """Compute name, normalized atomic ratios, and key properties from a dataframe row."""

    # unique name combining different characteristics of feedstock
    cropType = row["CropType"]
    name = (
        cropType
        + "_HHV" + str(np.round(row["HHV, MJ/kg"], 5))
        + "_Moisture" + str(np.round(row["Moisture content"], 5))
        + "_Ash" + str(np.round(row["Ash content"], 5))
    )

    ## get compositions in % mass and other properties
    # C,H,O
    wC = row["Carbon"] / 100
    wH = row["Hydrogen"] / 100
    wO = row["Oxygen"] / 100
    # impurities
    wN = np.float64(row["Nitrogen"]) / 100 * includeNS_ON
    wS = row["Sulfur"] / 100 * includeNS_ON

    HHV = row["HHV, MJ/kg"]
    if fixedMoisture_ON:
            moisture = fixedMoisture
    else:
        moisture = row["Moisture content"] / 100 * includeMoisture_ON
    ash = row["Ash content"] / 100 * includeAsh_ON

    # atomic ratios
    nC = wC / MW_C
    nH = wH / MW_H
    nO = wO / MW_O
    nN = wN / MW_N
    nS = wS / MW_S

    # convert to atomic ratios and normalize to carbon
    c = 1
    h = nH / nC
    o = nO / nC
    n = nN / nC
    s = nS / nC

    # return data as dict
    return {
        "name": name,
        "cropType": cropType,
        "c": c, "h": h, "o": o, "n": n, "s": s,
        "HHV": HHV,
        "moisture": moisture,
        "ash": ash,
    }




####################################################################
### loop over biomass sources and get adiabatic temp
# create T_array, in degrees celcius
names_array = []
cropType_array = []
excess_array = []
moisture_array = []
ash_array = []
T_array = []
Tair_array = []
composition_array = []
reaction_array = []
biomass_array = []
outlet_array = []
CO_to_COx_array = []
NO2_to_N_array = []

# include N/S
includeNS_ON = 1
includeMoisture_ON = 1
fixedMoisture_ON = 0
includeAsh_ON = 1
# define excess air
excess = 30 / 100 # [x / 100 for x in [0, 50, 100]]
# CO/COx ratio
CO_to_COx = 0 / 100
# NO2/N ratio
NO2_to_N  = 10 / 100
# heat loss
heat_loss = 0 / 100
# preheat temperature
T_preheat = 25 + 273.15


### RUN basecase
# loop
for index, row in df.iterrows():

    # if index !=0:# != 349/359:
    #     continue

    try:
        # initialize composition
        comp = initializeComposition(row, includeNS_ON, includeMoisture_ON, includeAsh_ON)

        ### run combustion function
        T_adiabatic, reaction_loc, Biomass_loc, outlet_loc = getFlueGasTemperature(comp["c"],comp["h"],comp["o"],comp["n"],comp["s"],comp["ash"],comp["moisture"],comp["HHV"],excess,CO_to_COx,NO2_to_N,heat_loss,T_preheat,comp["name"])

        # store
        names_array.append(comp["name"])
        cropType_array.append(comp["cropType"])
        CO_to_COx_array.append(CO_to_COx * 100)
        NO2_to_N_array.append(NO2_to_N * 100)
        excess_array.append(excess * 100)
        moisture_array.append(comp["moisture"] * 100)
        ash_array.append(comp["ash"] * 100)
        T_array.append(T_adiabatic - 273.15)
        Tair_array.append(T_preheat - 273.15)
        composition_array.append( (comp["h"],comp["o"]) )
        reaction_array.append(reaction_loc)
        biomass_array.append(Biomass_loc)
        outlet_array.append(outlet_loc)

    except:
        print("Error in calculation")
        print(row)
        print(" ")


## print
print("")
print("Minimum temperature, °C:")
print(np.min(T_array))
print("")
print("Median temperature, °C:")
print(np.median(T_array))
print("")
print("Maximum temperature, °C:")
print(np.max(T_array))
print("")


## create df for results
df_outputs = pd.DataFrame({"Name":names_array,
                         "Crop Type":cropType_array,
                         "Formula":np.array([t.formula for t in biomass_array]),
                         "Excess Air": excess_array,
                         "Ash": ash_array,
                         "Moisture": moisture_array,
                         "HHV": [t.HHV / (1e6 / 1e3 * t.MW) for t in biomass_array],
                         "H/C": [t[0] for t in composition_array],
                         "O/C": [t[1] for t in composition_array],
                         "Adiabatic Temperature, °C":T_array,
                         "Outlet, CO2 [mol %]":np.array([s.get_molar_fraction(('CO2')) for s in outlet_array]) * 100,
                         "Outlet, NO2 [ppm]":np.array([s.get_molar_fraction(('NO2')) for s in outlet_array])   * 1e6,
                         "Outlet, SO2 [ppm]":np.array([s.get_molar_fraction(('SO2')) for s in outlet_array])   * 1e6,
                         "Outlet, CO [mol %]":np.array([s.get_molar_fraction(('CO')) for s in outlet_array]) * 100,
                         "Outlet, H2O [mol %]":np.array([s.get_molar_fraction(('H2O')) for s in outlet_array]) * 100,
                         "Outlet, N2 [mol %]":np.array([s.get_molar_fraction(('N2')) for s in outlet_array]) * 100,
                         "Outlet, O2 [mol %]":np.array([s.get_molar_fraction(('O2')) for s in outlet_array]) * 100,
                         })
df_lowTemp = df_outputs[df_outputs["Adiabatic Temperature, °C"] < 500]



### base case ends here
####################################################################
####################################################################
### output data for manuscript
from openpyxl.utils import get_column_letter
from openpyxl import load_workbook

# save inputs
df_inputs_final  = df.copy()
df_outputs_final = df_outputs.copy()

# add units
df_outputs.rename(columns={
    'Excess Air': 'Excess Air [%]',
    'Ash': 'Ash [%_db]',
    'Moisture': 'Moisture [%_ar]',
    'HHV': 'HHV [MJ/kg_daf]'
}, inplace=True)

df_inputs_final.rename(columns={
    'HHV, MJ/kg': 'HHV [MJ/kg_daf]',
    'Ash content': 'Ash [%_db]',
    'Moisture content': 'Moisture [%_ar]',
    'Carbon': 'Carbon [%_daf]',
    'Oxygen': 'Oxygen [%_daf]',
    'Hydrogen': 'Hydrogen [%_daf]',
    'Nitrogen': 'Nitrogen [%_daf]',
    'Sulfur': 'Sulfur [%_daf]',
}, inplace=True)


filenameOutput = "Outputs/" + "biomassData"


df_sheets = {
    'Feedstock Input Data': df_inputs_final,
    'Adiabatic Combustion Results': df_outputs_final
}

with pd.ExcelWriter(filenameOutput + ".xlsx", engine='openpyxl') as writer:
    df_inputs_final.to_excel(writer, sheet_name='Feedstock Input Data', index=False)
    df_outputs_final.to_excel(writer, sheet_name='Adiabatic Combustion Results', index=False)
# 


# Adjust column widths
workbook = load_workbook(filenameOutput + ".xlsx")

for sheet_name, df_loc in df_sheets.items():
    sheet = workbook[sheet_name]

    # Apply auto-filter
    # Default: use header row (row 1)
    sheet.auto_filter.ref = sheet.dimensions

    # Freeze header row
    sheet.freeze_panes = "A2"

        
    # Loop through columns
    for col in sheet.columns:
        max_length = 0

        try:
            # Get column letter
            column_letter = col[0].column_letter
        except:
            column_letter = col[1].column_letter

        for cell in col:
            try:
                cell_value = str(cell.value) if cell.value is not None else ""
                length = len(cell_value)
                if length > max_length:
                    max_length = length
            except:
                pass

        # Add padding
        sheet.column_dimensions[column_letter].width = max_length * 1.05

# Save the adjusted Excel file
workbook.save(filenameOutput + ".xlsx")
#####################################################


####################################################################
####################################################################
### VALIDATION
validationCase_ON = 0
if validationCase_ON:
    #####
    ### validation, coal
    c = 1
    h = 0.94
    o = 0.32
    n = 0.012
    s = 0.0023
    #
    moisture = 10.8 / 100
    ash = 5.68 / 100
    HHV_ar = 26.535
    HHV_loc = HHV_ar / (1 - moisture - ash)
    ash = ash / (1-moisture)
    print("HHV: "+str(np.round(HHV_loc,2)))
    T_adiabatic, _, _, _ = getFlueGasTemperature(c,h,o,n,s,ash,moisture,HHV_loc,0,CO_to_COx,NO2_to_N,heat_loss,T_preheat,"")
    print(np.round(T_adiabatic,2))


    ### validation, manure
    c = 1
    h = 1.37
    o = 0.57
    n = 0.065
    s = 0.01
    #
    moisture = 36.61 / 100
    ash = 25.25 / 100
    HHV_ar = 7865 / 1e3
    HHV_loc = HHV_ar / (1 - moisture - ash)
    ash = ash / (1-moisture)
    # print("HHV: "+str(np.round(HHV_loc,2)))
    T_adiabatic, _, _, _ = getFlueGasTemperature(c,h,o,n,s,ash,moisture,HHV_loc,0,CO_to_COx,NO2_to_N,heat_loss,T_preheat,"")
    print(np.round(T_adiabatic,2))


    ### validation, partially dec manure
    c = 1
    h = 1.32
    o = 0.596
    n = 0.077
    s = 0.0132
    #
    moisture = 30.02 / 100
    ash = 28.01 / 100
    HHV_ar = 8305 / 1e3
    HHV_loc = HHV_ar / (1 - moisture - ash)
    ash = ash / (1-moisture)
    # print("HHV: "+str(np.round(HHV_loc,2)))
    T_adiabatic, _, _, _ = getFlueGasTemperature(c,h,o,n,s,ash,moisture,HHV_loc,0,CO_to_COx,NO2_to_N,heat_loss,T_preheat,"")
    print(np.round(T_adiabatic,2))


    ### validation, composted manure
    c = 1
    h = 1.23
    o = 0.585
    n = 0.0937
    s = 0.019
    #
    moisture = 35.35 / 100
    ash = 30.73 / 100
    HHV_ar = 6610 / 1e3
    HHV_loc = HHV_ar / (1 - moisture - ash)
    ash = ash / (1-moisture)
    # print("HHV: "+str(np.round(HHV_loc,2)))
    T_adiabatic, _, _, _ = getFlueGasTemperature(c,h,o,n,s,ash,moisture,HHV_loc,0,CO_to_COx,NO2_to_N,heat_loss,T_preheat,"")
    print(np.round(T_adiabatic,2))

