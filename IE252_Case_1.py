import pyomo.environ as pyo
from pyomo.opt import SolverFactory

set_i = [1,2,3,4]
set_j = [1,2]
set_k = [1,2,3]
set_m = [1,2,3,4,5]
set_n = [3,4]

dict_HP = {1:15, 2:12, 3:20, 4:19}
dict_HCW = {1:7, 2:5}

dict_KPR = {1:7000}

dict_CPR = {1:35}
dict_KP = {1:1800, 2:1400, 3:1900, 4:1600}
dict_CP = {1:56, 2:42, 3:35, 4:49}
dict_KAS = {3:3450, 4:3200}
dict_CAS = {3:21, 4:35}
dict_KT = {1:1750, 2:1550, 3:1800, 4:1450}
dict_CT = {1:28, 2:35, 3:21, 4:42}
dict_KPA = {1:1700, 2:1200, 3:2000, 4:1400}
dict_CPA = {1:14, 2:21, 3:28, 4:35}
dict_CSR = {1:15, 2:12}
dict_KSR = {1:3500, 2:2800}
dict_KIP = {1:100, 2:80, 3:100, 4:90}
dict_KIW = {1:2700, 2:2000}

mdl = pyo.ConcreteModel('Case_1')

mdl.I = pyo.Set(initialize=set_i, doc="plants")
mdl.I2 = pyo.Set(initialize=set_i, doc="plants")
mdl.J = pyo.Set(initialize=set_j, doc="Warehouses")
mdl.K = pyo.Set(initialize=set_k, doc="Distrubition Points")
mdl.L = pyo.Set(initialize=set_n, doc="assembly plants")
mdl.M = pyo.Set(initialize=set_m, doc="production processes")
mdl.N = pyo.Set(initialize=set_n, doc="for movements in plants")
mdl.Q = pyo.Set(initialize=set_j, doc="quarters")

mdl.HP = pyo.Param(mdl.I, initialize=dict_HP, doc="Holding costs for each plant")
mdl.HCW = pyo.Param(mdl.J, initialize=dict_HCW, doc="Holding costs for each warehouse")

mdl.KPR = pyo.Param({1}, initialize=dict_KPR, doc="procurement capacity of Ankara plant")
mdl.CPR = pyo.Param({1}, initialize=dict_CPR, doc="procurement cost of Ankara plant")
mdl.KP = pyo.Param(mdl.I, initialize=dict_KP, doc="process capacity of each plant")
mdl.CP = pyo.Param(mdl.I, initialize=dict_CP, doc="process cost of each plant")
mdl.KAS = pyo.Param(mdl.L, initialize=dict_KAS, doc="assembly capacity of İstanbul and Antalya, respectively")
mdl.CAS = pyo.Param(mdl.L, initialize=dict_CAS, doc="assembly cost of İstanbul and Antalya, respectively")
mdl.KTE = pyo.Param(mdl.I, initialize=dict_KT, doc="testing capacity of each plant")
mdl.CTE = pyo.Param(mdl.I, initialize=dict_CT, doc="testing cost of each plant")
mdl.KPA = pyo.Param(mdl.I, initialize=dict_KPA, doc="packaging capacity of each plant")
mdl.CPA = pyo.Param(mdl.I, initialize=dict_CPA, doc="packaging cost of each plant")
mdl.KIP = pyo.Param(mdl.I, initialize=dict_KIP, doc="Inventory holding capacity of each plant")
mdl.KIW = pyo.Param(mdl.J, initialize=dict_KIW, doc="Inventory holding capacity of each warehouse")
mdl.CSR = pyo.Param(mdl.J, initialize=dict_CSR, doc="sorting costs of warehouses")
mdl.KSR = pyo.Param(mdl.J, initialize=dict_KSR, doc="sorting capacity of warehouses")

mdl.vPR = pyo.Var({1}, mdl.Q, doc='amount of raw material procured at Ankara', within=pyo.NonNegativeReals)
mdl.vP = pyo.Var(mdl.I, mdl.Q, doc="amount of product processed at plants", within=pyo.NonNegativeReals)
mdl.vAS = pyo.Var(mdl.L, mdl.Q, doc="amount of product assembled at plants İstanbul and Antalya, respectively", 
                  within=pyo.NonNegativeReals)
mdl.vTE = pyo.Var(mdl.I, mdl.Q, doc="amount of product tested at plants", within=pyo.NonNegativeReals)
mdl.vPA = pyo.Var(mdl.I, mdl.Q, doc="amount of product packaged at plants", within=pyo.NonNegativeReals)

mdl.vOP = pyo.Var(mdl.I, doc="amount of product held at plants inventory in the first quarter", within=pyo.NonNegativeReals)
mdl.vOCW = pyo.Var(mdl.J, doc="amount of product held at warehouses inventory in the first quarter", within=pyo.NonNegativeReals)
mdl.vH = pyo.Var(mdl.J, mdl.Q, doc="amount of product sorted in warehouse in each quarter", within=pyo.NonNegativeReals)

mdl.vS = pyo.Var(mdl.K, mdl.Q, doc="amount of total supply product shipped to distribution center in each quarter",
                 within=pyo.NonNegativeReals)

mdl.vY = pyo.Var(mdl.I, mdl.J, mdl.Q, doc="amount of product shipped from plant to warehouse in each quarter", 
                 within=pyo.NonNegativeReals)

mdl.vW = pyo.Var(mdl.J, mdl.K, mdl.Q, doc="amount of product shipped from warehouse to distribution center in each quarter", 
                 within=pyo.NonNegativeReals)

demand1 = {1:2000, 2:1700, 3:1800}
demand2 = {1:2500, 2:1500, 3:2050}

W_to_DC_1 = {1:202, 2:561, 3:867}
W_to_DC_2 = {1:817, 2:304, 3:453}
P_to_W_1 = {1:589, 2:417, 3:479, 4:454}
P_to_W_2 = {1:346, 2:512, 3:774, 4:573}

mdl.D1 = pyo.Param(mdl.K, initialize=demand1, doc="demand for each distribution center in first quarter")
mdl.D2 = pyo.Param(mdl.K, initialize=demand2, doc="demand for each distribution center in second quarter")

mdl.B1 = pyo.Param(mdl.K, initialize=W_to_DC_1, doc="shipment cost from İzmir warehouse to distribution centers")
mdl.B2 = pyo.Param(mdl.K, initialize=W_to_DC_2, doc="shipment cost from Kayseri warehouse to distribution centers")

mdl.A1 = pyo.Param(mdl.I, initialize=P_to_W_1, doc="shipment cost from plants to İzmir warehouse")
mdl.A2 = pyo.Param(mdl.I, initialize=P_to_W_2, doc="shipment cost from plants to Kayseri warehouse")


dict_Ankara = {1:0, 2:235, 3:445, 4:476}
dict_Eskisehir = {1:235, 2:0, 3:303, 4:415}
dict_istanbul = {1:445, 2:303, 3:0, 4:695}
dict_Antalya = {1:476, 2:415, 3:695, 4:0}

mdl.T1 = pyo.Param(mdl.I, initialize=dict_Ankara, doc="distance (km) between Ankara and other plants")
mdl.T2 = pyo.Param(mdl.I, initialize=dict_Eskisehir, doc="distance (km) between Eskişehir and other plants")
mdl.T3 = pyo.Param(mdl.I, initialize=dict_istanbul, doc="distance (km) between İstanbul and other plants")
mdl.T4 = pyo.Param(mdl.I, initialize=dict_Antalya, doc="distance (km) between Antalya and other plants")

mdl.vT = pyo.Var(mdl.M, mdl.I, mdl.I2, mdl.Q, doc="amount of unfinished product that transported between plants in each quarter", 
                 within=pyo.NonNegativeReals)

mdl.vBO = pyo.Var(mdl.K, doc="backorder if a retailer DC's demand can't be satisfied", within=pyo.NonNegativeReals)
mdl.CBO = pyo.Param({1},initialize={1:37})

def procurement_capacity_constraint(mdl,q):
    return mdl.vPR[1,q] <= mdl.KPR[1]
mdl.procurement_capacity_constraint = pyo.Constraint(mdl.Q, rule=procurement_capacity_constraint)

def process_capacity_constraint(mdl, i, q):
    return mdl.vP[i,q] <= mdl.KP[i]
mdl.process_capacity_constraint = pyo.Constraint(mdl.I, mdl.Q, rule=process_capacity_constraint)

def assembly_capacity_constraint(mdl, l, q):
    return mdl.vAS[l,q] <= mdl.KAS[l]
mdl.assembly_capacity_constraint = pyo.Constraint(mdl.L, mdl.Q, rule=assembly_capacity_constraint)

def testing_capacity_constraint(mdl, i, q):
    return mdl.vTE[i,q] <= mdl.KTE[i]
mdl.testing_capacity_constraint = pyo.Constraint(mdl.I, mdl.Q, rule=testing_capacity_constraint)

def packaging_capacity_constraint(mdl, i, q):
    return mdl.vPA[i,q] <= mdl.KPA[i]
mdl.packaging_capacity_constraint = pyo.Constraint(mdl.I, mdl.Q, rule=packaging_capacity_constraint)

def plant_inventory_capacity_constraint(mdl, i):
    return mdl.vOP[i] <= mdl.KIP[i]
mdl.plant_inventory_capacity_constraint = pyo.Constraint(mdl.I, rule=plant_inventory_capacity_constraint)

def warehouse_inventory_capacity_constraint(mdl, j):
    return mdl.vOCW[j] <= mdl.KIW[j]
mdl.warehouse_inventory_capacity_constraint = pyo.Constraint(mdl.J, rule=warehouse_inventory_capacity_constraint)

def sorting_capacity_constraint(mdl, j, q):
    return mdl.vH[j,q] <= mdl.KSR[j]
mdl.sorting_capacity_constraint = pyo.Constraint(mdl.J, mdl.Q, rule=sorting_capacity_constraint)

def sorting_constraint_1(mdl,i,j):
    return mdl.vH[j,1] + mdl.vOCW[j] == sum(mdl.vY[i,j,1] for i in mdl.I)
mdl.sorting_constraint_1 = pyo.Constraint(mdl.I, mdl.J, rule=sorting_constraint_1)

def sorting_constraint_2(mdl,i,j):
    return mdl.vH[j,2] == sum(mdl.vY[i,j,2] for i in mdl.I) + mdl.vOCW[j]
mdl.sorting_constraint_2 = pyo.Constraint(mdl.I, mdl.J, rule=sorting_constraint_2)

def sorting_constraint_3(mdl,j,k,q):
    return mdl.vH[j,q] == sum(mdl.vW[j,k,q] for k in mdl.K)
mdl.sorting_constraint_3 = pyo.Constraint(mdl.J, mdl.K, mdl.Q, rule=sorting_constraint_3)


def demand_constraint_1(mdl, j,k):
    return mdl.D1[k] <= sum(mdl.vW[j,k,1] for j in mdl.J) + mdl.vBO[k]
mdl.demand_constraint_1 = pyo.Constraint(mdl.J, mdl.K, rule=demand_constraint_1)

def demand_constraint_2(mdl, j,k):
    return mdl.D2[k] + mdl.vBO[k] <= sum(mdl.vW[j,k,2] for j in mdl.J)
mdl.demand_constraint_2 = pyo.Constraint(mdl.J, mdl.K, rule=demand_constraint_2)


def flow_constraint_3(mdl,i,j):
    return sum(mdl.vY[i,j,1] for j in mdl.J) + mdl.vOP[i] == mdl.vPA[i,1]
mdl.flow_constraint_3 = pyo.Constraint(mdl.I, mdl.J, rule=flow_constraint_3)

def flow_constraint_4(mdl,i,j):
    return sum(mdl.vY[i,j,2]for j in mdl.J) == mdl.vPA[i,2] + mdl.vOP[i]
mdl.flow_constraint_4 = pyo.Constraint(mdl.I, mdl.J, rule=flow_constraint_4)

def plant_flow_constraint_1(mdl,i,q):
    return mdl.vPR[1,q] == sum(mdl.vP[i,q] for i in mdl.I)
mdl.plant_flow_constraint_1 = pyo.Constraint(mdl.I, mdl.Q, rule=plant_flow_constraint_1)

def plant_flow_constraint_2(mdl,i,l,q):
    return sum(mdl.vP[i,q] for i in mdl.I) == sum(mdl.vAS[l,q] for l in mdl.L)
mdl.plant_flow_constraint_2 = pyo.Constraint(mdl.I, mdl.L, mdl.Q, rule=plant_flow_constraint_2)

def plant_flow_constraint_3(mdl,i,l,q):
    return sum(mdl.vAS[l,q] for l in mdl.L) == sum(mdl.vTE[i,q] for i in mdl.I)
mdl.plant_flow_constraint_3 = pyo.Constraint(mdl.I, mdl.L, mdl.Q, rule=plant_flow_constraint_3)

def plant_flow_constraint_4(mdl,i,q):
    return sum(mdl.vTE[i,q] for i in mdl.I) == sum(mdl.vPA[i,q] for i in mdl.I)
mdl.plant_flow_constraint_4 = pyo.Constraint(mdl.I, mdl.Q, rule=plant_flow_constraint_4)

def transport_constraint_5(mdl,i,q):
    return mdl.vT[1,1,i,q] == mdl.vP[i,q]
mdl.transport_constraint_5 = pyo.Constraint(mdl.I, mdl.Q, rule=transport_constraint_5)

def transport_constraint_6(mdl,i,l,q):
    return sum(mdl.vT[2,i,l,q] for l in mdl.L) == mdl.vP[i,q]
mdl.transport_constraint_6 = pyo.Constraint(mdl.I, mdl.L, mdl.Q, rule=transport_constraint_6)

def transport_constraint_7(mdl,i,l,q):
    return sum(mdl.vT[2,i,l,q] for i in mdl.I) == mdl.vAS[l,q]
mdl.transport_constraint_7 = pyo.Constraint(mdl.I, mdl.L, mdl.Q, rule=transport_constraint_7)

def transport_constraint_8(mdl,i,l,q):
    return sum(mdl.vT[3,l,i,q] for i in mdl.I) == mdl.vAS[l,q]
mdl.transport_constraint_8 = pyo.Constraint(mdl.I, mdl.L, mdl.Q, rule=transport_constraint_8)

def transport_constraint_9(mdl,i,l,q):
    return sum(mdl.vT[3,l,i,q] for l in mdl.L) == mdl.vTE[i,q]
mdl.transport_constraint_9 = pyo.Constraint(mdl.I, mdl.L, mdl.Q, rule=transport_constraint_9)

def transport_constraint_20(mdl,i,i2,q):
    return sum(mdl.vT[4,i,i2,q] for i in mdl.I) == mdl.vPA[i2,q]
mdl.transport_constraint_20 = pyo.Constraint(mdl.I, mdl.I2, mdl.Q, rule=transport_constraint_20)

def transport_constraint_21(mdl,i,i2,q):
    return sum(mdl.vT[4,i,i2,q] for i2 in mdl.I2) == mdl.vTE[i,q] 
mdl.transport_constraint_21 = pyo.Constraint(mdl.I, mdl.I2, mdl.Q, rule=transport_constraint_21)



def distribution_constraint_2(mdl,j,k,q):
    return mdl.vS[k,q] == sum(mdl.vW[j,k,q] for j in mdl.J) 
mdl.distribution_constraint_2 = pyo.Constraint(mdl.J, mdl.K, mdl.Q, rule=distribution_constraint_2)

plants_production_costs = mdl.CPR[1]*(mdl.vPR[1,1] + mdl.vPR[1,2])
plants_production_costs += sum(mdl.CP[i] * sum(mdl.vP[i,q] for q in mdl.Q) for i in mdl.I)
plants_production_costs += sum(mdl.CAS[l] * sum(mdl.vAS[l,q] for q in mdl.Q) for l in mdl.L)
plants_production_costs += sum(mdl.CTE[i] * sum(mdl.vTE[i,q] for q in mdl.Q) for i in mdl.I)
plants_production_costs += sum(mdl.CPA[i] * sum(mdl.vPA[i,q] for q in mdl.Q) for i in mdl.I)

transportation_costs_p_to_w = 1.5 * sum(mdl.A1[i] * mdl.vY[i,1,1] for i in mdl.I)
transportation_costs_p_to_w += 1.5 * 1.4 * sum(mdl.A1[i] * mdl.vY[i,1,2] for i in mdl.I)
transportation_costs_p_to_w += 1.5 * sum(mdl.A2[i] * mdl.vY[i,2,1] for i in mdl.I)
transportation_costs_p_to_w += 1.5 * 1.4 * sum(mdl.A2[i] * mdl.vY[i,2,2] for i in mdl.I)
    
transportation_costs_w_to_dc = 1.5 * sum(mdl.B1[k] * mdl.vW[1,k,1] for k in mdl.K)
transportation_costs_w_to_dc += 1.5 * 1.4 * sum(mdl.B1[k] * mdl.vW[1,k,2] for k in mdl.K)
transportation_costs_w_to_dc += 1.5 * sum(mdl.B2[k] * mdl.vW[2,k,1] for k in mdl.K)
transportation_costs_w_to_dc += 1.5 * 1.4 * sum(mdl.B2[k] * mdl.vW[2,k,2] for k in mdl.K)

sorting_costs = sum(mdl.CSR[j] * sum(mdl.vH[j,q] for q in mdl.Q) for j in mdl.J)
    
inventory_holding_costs = sum(mdl.HP[i] * mdl.vOP[i] for i in mdl.I)
inventory_holding_costs += sum(mdl.HCW[j] * mdl.vOCW[j] for j in mdl.J)

plants_transportation_costs = 2 * sum(mdl.T1[i] * mdl.vT[1,1,i,1] for i in mdl.I)
plants_transportation_costs += 1.75 * sum(mdl.T1[i] * mdl.vT[2,1,i,1] for i in mdl.I)
plants_transportation_costs += 1.5 * sum(mdl.T1[i] * mdl.vT[4,1,i,1] for i in mdl.I)
plants_transportation_costs += 2 * 1.4 * sum(mdl.T1[i] * mdl.vT[1,1,i,2] for i in mdl.I)
plants_transportation_costs += 1.75 * 1.4 * sum(mdl.T1[i] * mdl.vT[2,1,i,2] for i in mdl.I)
plants_transportation_costs += 1.5 * 1.4 * sum(mdl.T1[i] * mdl.vT[4,1,i,2] for i in mdl.I)
    
plants_transportation_costs += 1.75 * sum(mdl.T2[i] * mdl.vT[2,2,i,1] for i in mdl.I)
plants_transportation_costs += 1.5 * sum(mdl.T2[i] * mdl.vT[4,2,i,1] for i in mdl.I)
plants_transportation_costs += 1.75 * 1.4 * sum(mdl.T2[i] * mdl.vT[2,2,i,2] for i in mdl.I)
plants_transportation_costs += 1.5 * 1.4 * sum(mdl.T2[i] * mdl.vT[4,2,i,2] for i in mdl.I)
    
plants_transportation_costs += 1.75 * sum(mdl.T3[i] * mdl.vT[2,3,i,1] for i in mdl.I)
plants_transportation_costs += 1.5 * sum(mdl.T3[i] * mdl.vT[3,3,i,1] for i in mdl.I)
plants_transportation_costs += 1.5 * sum(mdl.T3[i] * mdl.vT[4,3,i,1] for i in mdl.I)
plants_transportation_costs += 1.75 * 1.4 * sum(mdl.T3[i] * mdl.vT[2,3,i,2] for i in mdl.I)
plants_transportation_costs += 1.5 * 1.4 * sum(mdl.T3[i] * mdl.vT[3,3,i,2] for i in mdl.I)
plants_transportation_costs += 1.5 * 1.4 * sum(mdl.T3[i] * mdl.vT[4,3,i,2] for i in mdl.I)

plants_transportation_costs += 1.75 * sum(mdl.T4[i] * mdl.vT[2,4,i,1] for i in mdl.I)    
plants_transportation_costs += 1.5 * sum(mdl.T4[i] * mdl.vT[3,4,i,1] for i in mdl.I)
plants_transportation_costs += 1.5 * sum(mdl.T4[i] * mdl.vT[4,4,i,1] for i in mdl.I)
plants_transportation_costs += 1.75 * 1.4 * sum(mdl.T4[i] * mdl.vT[2,4,i,2] for i in mdl.I)
plants_transportation_costs += 1.5 * 1.4 * sum(mdl.T4[i] * mdl.vT[3,4,i,2] for i in mdl.I)
plants_transportation_costs += 1.5 * 1.4 * sum(mdl.T4[i] * mdl.vT[4,4,i,2] for i in mdl.I)

backorder_cost = mdl.CBO[1]* sum(mdl.vBO[k] for k in mdl.K)

all_costs = plants_production_costs + transportation_costs_p_to_w + transportation_costs_w_to_dc + sorting_costs + inventory_holding_costs + plants_transportation_costs + backorder_cost

mdl.obj_func = pyo.Objective(expr=all_costs, sense=pyo.minimize)

mdl.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT) #shadow prices of the constraints
mdl.rc = pyo.Suffix(direction=pyo.Suffix.IMPORT) #reduced costs of the objective function coefficients

Solver = SolverFactory('glpk')
SolverResults = Solver.solve(mdl)
SolverResults.write()
#(optional) Writing model declarations on the console via .pprint()
mdl.pprint()
#(optional) Exporting the open form of the model to file "mdl.lp" via .write(...)
mdl.write('mdl.lp', io_options={'symbolic_solver_labels': True})

mdl.vT.display()
mdl.vY.display()
mdl.vW.display()
mdl.obj_func.display()

vPR_dict = mdl.vPR.extract_values()
vP_dict = mdl.vP.extract_values()
vAS_dict = mdl.vAS.extract_values()
vTE_dict = mdl.vTE.extract_values()
vPA_dict = mdl.vPA.extract_values()
OP_dict = mdl.vOP.extract_values()
OCW_dict = mdl.vOCW.extract_values()
h_dict = mdl.vH.extract_values()
T_dict = mdl.vT.extract_values()
Y_dict = mdl.vY.extract_values()
W_dict = mdl.vW.extract_values()
S_dict = mdl.vS.extract_values()
opt_z = pyo.value(mdl.obj_func)

print(opt_z)




