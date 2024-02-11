import numpy as np
import pandas as pd
import pyomo.environ as pyo
from pyomo.opt import SolverFactory


#Constructing the model object as "mdl"
mdl = pyo.ConcreteModel('Case_2')

project_a = [(70,42) , (70,42) , (126,77) , (105,63) , (35,21) , (35,21) , (105,63) , (119,70),
       (35,21) , (35,21) , (70,42) , (49,28) , (35,21) , (35,21) , (28,21)]

project_b = [(139,81),(78,46),(292,170),(34,20),(90,53),(18,11),(48,28),(24,14),(72,42),(19,11),
       (36,21),(17,10),(96,56),(96,56)]

set_i = [1,2]
set_j = [x for x in range(len(project_a))]
set_k = [x for x in range(len(project_b))]
mdl.I = pyo.Set(initialize=set_i, doc="project type")
mdl.J = pyo.Set(initialize=set_j, doc="project A WBS")
mdl.K = pyo.Set(initialize=set_k, doc="project B WBS")

dic_p = {1:220/3, 2:250/3}
dic_u = {1:253/3, 2:287.5/3}

# wage rates of first year is P, second year is U. i represents the project type
mdl.P = pyo.Param(mdl.I, initialize=dic_p, doc="wage rates of first year")
mdl.U = pyo.Param(mdl.I, initialize=dic_u, doc="wage rates of second year")

dict_a1 = {x:project_a[x][0] for x in range(len(project_a))}
dict_a2 = {x:project_a[x][1] for x in range(len(project_a))}
dict_b1 = {x:project_b[x][0] for x in range(len(project_b))}
dict_b2 = {x:project_b[x][1] for x in range(len(project_b))}

# parameters of durations of each node WBS
mdl.da_max = pyo.Param(mdl.J, initialize=dict_a1, doc="max durations of each project A WBS")
mdl.da_min = pyo.Param(mdl.J, initialize=dict_a2, doc="min durations of each project A WBS")
mdl.db_max = pyo.Param(mdl.K, initialize=dict_b1, doc="max durations of each project B WBS")
mdl.db_min = pyo.Param(mdl.K, initialize=dict_b2, doc="min durations of each project B WBS")

dict_worker = {1:40, 2:60}
mdl.W = pyo.Param(mdl.I, initialize=dict_worker, doc="number of workers initially assigned to projects")

mdl.C = pyo.Var(mdl.I, within=pyo.NonNegativeReals, doc="Number of type c workers allocated to project type i")
mdl.D_a = pyo.Var(mdl.J, within=pyo.NonNegativeReals, doc="Duration of each WBS in project A")
mdl.D_b = pyo.Var(mdl.K, within=pyo.NonNegativeReals, doc="Duration of each WBS in project B")
mdl.S_a = pyo.Var(mdl.J, within=pyo.NonNegativeReals, doc="Starting time of each WBS in project A")
mdl.S_b = pyo.Var(mdl.K, within=pyo.NonNegativeReals, doc="Starting time of each WBS in project B")
mdl.F_a = pyo.Var(mdl.J, within=pyo.NonNegativeReals, doc="Finishing time of each WBS in project A")
mdl.F_b = pyo.Var(mdl.K, within=pyo.NonNegativeReals, doc="Finishing time of each WBS in project B")
mdl.A = pyo.Var(within=pyo.NonNegativeReals, doc="Finishing time of project A")
mdl.B = pyo.Var(within=pyo.NonNegativeReals, doc="Finishing time of project B")
# x: number of days in first year for each project
# y: number of days in second year for each project
mdl.x = pyo.Var(mdl.I, within=pyo.NonNegativeReals, doc="number of days in first year for each project")
mdl.y = pyo.Var(mdl.I, within=pyo.NonNegativeReals, doc="number of days in second year for each project")


def starting_constraint1(mdl):
    return mdl.S_a[0] == 0
mdl.starting_constraint1 = pyo.Constraint(rule=starting_constraint1, 
                                          doc='first node starts from zero')
def starting_constraint2(mdl):
    return mdl.S_b[0] == 0
mdl.starting_constraint2 = pyo.Constraint(rule=starting_constraint2, 
                                          doc='first node starts from zero')

def linear_interpolation_a(mdl, j):
    return mdl.D_a[j] == mdl.da_max[j] - ((mdl.da_max[j]-mdl.da_min[j])/100)*mdl.C[1]
mdl.linear_interpolation_a = pyo.Constraint(mdl.J, rule=linear_interpolation_a, 
                                            doc="finding the days in between with linear interpolation")

def linear_interpolation_b(mdl, k):
    return mdl.D_b[k] == mdl.db_max[k] - ((mdl.db_max[k]-mdl.db_min[k])/100)*mdl.C[2]
mdl.linear_interpolation_b = pyo.Constraint(mdl.K, rule=linear_interpolation_b, 
                                            doc="finding the days in between with linear interpolation")

def finishing_constraint_a(mdl, j):
    return  mdl.F_a[j] == mdl.S_a[j] + mdl.D_a[j]
mdl.finishing_constraint_a = pyo.Constraint(mdl.J, rule=finishing_constraint_a, 
                                          doc='finishing means starting plus duration')
def finishing_constraint_b(mdl, k):
    return  mdl.F_b[k] == mdl.S_b[k] + mdl.D_b[k]
mdl.finishing_constraint_b = pyo.Constraint(mdl.K, rule=finishing_constraint_b, 
                                          doc='finishing means starting plus duration')


def pro_cons_2a(mdl):
    return mdl.S_a[2-1] >= mdl.F_a[1-1]
mdl.pro_cons_2a = pyo.Constraint(rule=pro_cons_2a)

def pro_cons_3a(mdl):
    return mdl.S_a[3-1] >= mdl.F_a[2-1]
mdl.pro_cons_3a = pyo.Constraint(rule=pro_cons_3a)

def pro_cons_4a(mdl):
    return mdl.S_a[4-1] >= mdl.F_a[3-1]
mdl.pro_cons_4a = pyo.Constraint(rule=pro_cons_4a)

def pro_cons_5a(mdl):
    return mdl.S_a[5-1] >= mdl.F_a[1-1]
mdl.pro_cons_5a = pyo.Constraint(rule=pro_cons_5a)

def pro_cons_6a(mdl):
    return mdl.S_a[6-1] >= mdl.F_a[5-1]
mdl.pro_cons_6a = pyo.Constraint(rule=pro_cons_6a)

def pro_cons_7a(mdl):
    return mdl.S_a[7-1] >= mdl.F_a[6-1]
mdl.pro_cons_7a = pyo.Constraint(rule=pro_cons_7a)

def pro_cons_8a(mdl):
    return mdl.S_a[8-1] >= mdl.F_a[7-1]
mdl.pro_cons_8a = pyo.Constraint(rule=pro_cons_8a)

def pro_cons_9a(mdl):
    return mdl.S_a[9-1] >= mdl.F_a[8-1]
mdl.pro_cons_9 = pyo.Constraint(rule=pro_cons_9a)

def pro_cons_10a(mdl):
    return mdl.S_a[10-1] >= mdl.F_a[6-1]
mdl.pro_cons_10 = pyo.Constraint(rule=pro_cons_10a)

def pro_cons_11a(mdl):
    return mdl.S_a[11-1] >= mdl.F_a[10-1]
mdl.pro_cons_11a = pyo.Constraint(rule=pro_cons_11a)

def pro_cons_12a(mdl):
    return mdl.S_a[12-1] >= mdl.F_a[11-1]
mdl.pro_cons_12a = pyo.Constraint(rule=pro_cons_12a)

def pro_cons_13a(mdl):
    return mdl.S_a[13-1] >= mdl.F_a[11-1]
mdl.pro_cons_13a = pyo.Constraint(rule=pro_cons_13a)

def pro_cons_14a(mdl):
    return mdl.S_a[14-1] >= mdl.F_a[11-1]
mdl.pro_cons_14a = pyo.Constraint(rule=pro_cons_14a)

def pro_cons_15_1a(mdl):
    return mdl.S_a[15-1] >= mdl.F_a[9-1]
mdl.pro_cons_15_1a = pyo.Constraint(rule=pro_cons_15_1a)

def pro_cons_15_2a(mdl):
    return mdl.S_a[15-1] >= mdl.F_a[14-1]
mdl.pro_cons_15_2a = pyo.Constraint(rule=pro_cons_15_2a)

def final_1a(mdl):
    return mdl.A >= mdl.F_a[15-1]
mdl.final_1a = pyo.Constraint(rule=final_1a)

def final_2a(mdl):
    return mdl.A >= mdl.F_a[4-1]
mdl.final_2a = pyo.Constraint(rule=final_2a)

def final_3a(mdl):
    return mdl.A >= mdl.F_a[12-1]
mdl.final_3a = pyo.Constraint(rule=final_3a)

def final_4a(mdl):
    return mdl.A >= mdl.F_a[13-1]
mdl.final_4 = pyo.Constraint(rule=final_4a)

def pro_cons_2b(mdl):
    return mdl.S_b[2-1] >= mdl.F_b[1-1]
mdl.pro_cons_2b = pyo.Constraint(rule=pro_cons_2b)

def pro_cons_3b(mdl):
    return mdl.S_b[3-1] >= mdl.F_b[2-1]
mdl.pro_cons_3b = pyo.Constraint(rule=pro_cons_3b)

def pro_cons_4b(mdl):
    return mdl.S_b[4-1] >= mdl.F_b[3-1]
mdl.pro_cons_4b = pyo.Constraint(rule=pro_cons_4b)

def pro_cons_5b(mdl):
    return mdl.S_b[5-1] >= mdl.F_b[4-1]
mdl.pro_cons_5b = pyo.Constraint(rule=pro_cons_5b)

def pro_cons_6b(mdl):
    return mdl.S_b[6-1] >= mdl.F_b[5-1]
mdl.pro_cons_6b = pyo.Constraint(rule=pro_cons_6b)

def pro_cons_7b(mdl):
    return mdl.S_b[7-1] >= mdl.F_b[6-1]
mdl.pro_cons_7b = pyo.Constraint(rule=pro_cons_7b)

def pro_cons_8b(mdl):
    return mdl.S_b[8-1] >= mdl.F_b[7-1]
mdl.pro_cons_8b = pyo.Constraint(rule=pro_cons_8b)

def pro_cons_9b(mdl):
    return mdl.S_b[9-1] >= mdl.F_b[5-1]
mdl.pro_cons_9b = pyo.Constraint(rule=pro_cons_9b)

def pro_cons_10_1b(mdl):
    return mdl.S_b[10-1] >= mdl.F_b[8-1]
mdl.pro_cons_10_1b = pyo.Constraint(rule=pro_cons_10_1b)

def pro_cons_10_2b(mdl):
    return mdl.S_b[10-1] >= mdl.F_b[9-1]
mdl.pro_cons_10_2b = pyo.Constraint(rule=pro_cons_10_2b)

def pro_cons_11b(mdl):
    return mdl.S_b[11-1] >= mdl.F_b[10-1]
mdl.pro_cons_11b = pyo.Constraint(rule=pro_cons_11b)

def pro_cons_12b(mdl):
    return mdl.S_b[12-1] >= mdl.F_b[11-1]
mdl.pro_cons_12b = pyo.Constraint(rule=pro_cons_12b)

def pro_cons_13b(mdl):
    return mdl.S_b[13-1] >= mdl.F_b[5-1]
mdl.pro_cons_13b = pyo.Constraint(rule=pro_cons_13b)

def pro_cons_14_1b(mdl):
    return mdl.S_b[14-1] >= mdl.F_b[12-1]
mdl.pro_cons_14_1b = pyo.Constraint(rule=pro_cons_14_1b)

def pro_cons_14_2b(mdl):
    return mdl.S_b[14-1] >= mdl.F_b[13-1]
mdl.pro_cons_14_2b = pyo.Constraint(rule=pro_cons_14_2b)

def final(mdl):
    return mdl.B == mdl.F_b[14-1]
mdl.final = pyo.Constraint(rule=final)

def day_constraint1(mdl, i):
    return mdl.x[i] <= 360
mdl.day_constraint1 = pyo.Constraint(mdl.I, rule=day_constraint1, doc="number of days in first year cannot exceed 360")

def day_constraint2(mdl, i):
    return mdl.y[i] <= 360
mdl.day_constraint2 = pyo.Constraint(mdl.I, rule=day_constraint2, doc="number of days in second year cannot exceed 360")

def type_c_constraint(mdl):
    return mdl.C[1] + mdl.C[2] == 100
mdl.type_c_constraint = pyo.Constraint(rule=type_c_constraint, doc="total number of type c workers is 100")

def first_year_constraint(mdl):
    return mdl.A == mdl.x[1] + mdl.y[1]
mdl.first_year_constraint = pyo.Constraint(rule=first_year_constraint, doc="total number of days in project A")

def second_year_constraint(mdl):
    return mdl.B == mdl.x[2] + mdl.y[2]
mdl.second_year_constraint = pyo.Constraint(rule=second_year_constraint, doc="total number of days in project B")

def obj_func(mdl, i):
    return sum(mdl.W[i]*mdl.x[i]*mdl.P[i] + mdl.W[i]*mdl.y[i]*mdl.U[i]  for i in mdl.I)
mdl.obj_func = pyo.Objective(rule=obj_func, sense=pyo.minimize)

mdl.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT) #shadow prices of the constraints
mdl.rc = pyo.Suffix(direction=pyo.Suffix.IMPORT) #reduced costs of the objective function coefficients
#Specifying solver via SolverFactory(...) function in pyomo.opt module
Solver = SolverFactory('glpk')
SolverResults = Solver.solve(mdl)
SolverResults.write()
#(optional) Writing model declarations on the console via .pprint()
mdl.pprint()
#(optional) Exporting the open form of the model to file "mdl.lp" via .write(...)
mdl.write('mdl.lp', io_options={'symbolic_solver_labels': True})

#Reaching optimal values of variables and objective function on the console via .display()
mdl.A.display()
mdl.B.display()
mdl.obj_func.display()

#Storing optimal values of variables in a dictionary via .extract_values()
A_dict = mdl.A.extract_values()
B_dict = mdl.B.extract_values()
S_a_dict = mdl.S_a.extract_values()
S_b_dict = mdl.S_b.extract_values()
#Storing optimal objective function value via pyo.value(...)
opt_z = pyo.value(mdl.obj_func)


variable_data = {(i, v.name): pyo.value(v) for i,v in mdl.F_a.items()}
optimal_finishing_time_a = pd.DataFrame.from_dict(variable_data, orient="index", columns=["variable value F_a"])
optimal_finishing_time_a.to_excel(r'C:\Users\HP\Desktop\dersler\IE251\Case 2\Son\F_a.xlsx', sheet_name='Finishing Times a')

variable_data = {(i, v.name): pyo.value(v) for i,v in mdl.F_b.items()}
optimal_finishing_time_b = pd.DataFrame.from_dict(variable_data, orient="index", columns=["variable value F_b"])
optimal_finishing_time_b.to_excel(r'C:\Users\HP\Desktop\dersler\IE251\Case 2\Son\F_b.xlsx', sheet_name='Finishing Times b')

variable_data = {(i, v.name): pyo.value(v) for i,v in mdl.S_a.items()}
optimal_starting_time_a = pd.DataFrame.from_dict(variable_data, orient="index", columns=["variable value S_a"])
optimal_starting_time_a.to_excel(r'C:\Users\HP\Desktop\dersler\IE251\Case 2\Son\S_a.xlsx', sheet_name='Starting Times a')

variable_data = {(i, v.name): pyo.value(v) for i,v in mdl.S_b.items()}
optimal_starting_time_b = pd.DataFrame.from_dict(variable_data, orient="index", columns=["variable value S_b"])
optimal_starting_time_b.to_excel(r'C:\Users\HP\Desktop\dersler\IE251\Case 2\Son\S_b.xlsx', sheet_name='Starting Times a')

variable_data = {(i, v.name): pyo.value(v) for i,v in mdl.D_a.items()}
optimal_duration_time_a = pd.DataFrame.from_dict(variable_data, orient="index", columns=["variable value D_a"])
optimal_duration_time_a.to_excel(r'C:\Users\HP\Desktop\dersler\IE251\Case 2\Son\D_a.xlsx', sheet_name='Duration Times a')

variable_data = {(i, v.name): pyo.value(v) for i,v in mdl.D_b.items()}
optimal_duration_time_b = pd.DataFrame.from_dict(variable_data, orient="index", columns=["variable value D_b"])
optimal_duration_time_b.to_excel(r'C:\Users\HP\Desktop\dersler\IE251\Case 2\Son\D_b.xlsx', sheet_name='Duration Times a')


#export to excel: reduced costs
reduced_cost_dict={str(key):mdl.rc[key] for key in mdl.rc.keys()}
Reduced_Costs_print =pd.DataFrame.from_dict(reduced_cost_dict,orient="index", columns=["reduced cost"])
Reduced_Costs_print.to_excel(r'C:\Users\HP\Desktop\dersler\IE251\Case 2\Son\ReducedCostsPart1.xlsx', sheet_name='ReducedCosts')



#export to excel: shadow prices        
duals_dict = {str(key):mdl.dual[key] for key in mdl.dual.keys()}

u_slack_dict = {
    # uslacks for non-indexed constraints
    **{str(con):con.uslack() for con in mdl.component_objects(pyo.Constraint)
       if not con.is_indexed()},
    # indexed constraint uslack
    # loop through the indexed constraints
    # get all the indices then retrieve the slacks for each index of constraint
    **{k:v for con in mdl.component_objects(pyo.Constraint) if con.is_indexed()
       for k,v in {'{}[{}]'.format(str(con),key):con[key].uslack()
                   for key in con.keys()}.items()}
    }

l_slack_dict = {
    # lslacks for non-indexed constraints
    **{str(con):con.lslack() for con in mdl.component_objects(pyo.Constraint)
       if not con.is_indexed()},
    # indexed constraint lslack
    # loop through the indexed constraints
    # get all the indices then retrieve the slacks for each index of constraint
    **{k:v for con in mdl.component_objects(pyo.Constraint) if con.is_indexed()
       for k,v in {'{}[{}]'.format(str(con),key):con[key].lslack()
                   for key in con.keys()}.items()}
    }

# combine into a single df
Shadow_Prices_print = pd.concat([pd.Series(d,name=name)
           for name,d in {'duals':duals_dict,
                          'uslack':u_slack_dict,
                          'lslack':l_slack_dict}.items()],
          axis='columns')
Shadow_Prices_print.to_excel(r'C:\Users\HP\Desktop\dersler\IE251\Case 2\Son\ShadowPricesPart1.xlsx', sheet_name='ShadowPrices')
