import numpy as np
import pandas as pd
import pyomo.environ as pyo
from pyomo.opt import SolverFactory

people_data = pd.read_excel(r'C:\Users\HP\Desktop\dersler\IE252\Case 2\IE252 Case 2 Tables.xlsx',sheet_name='Table2')
skills_data = pd.read_excel(r'C:\Users\HP\Desktop\dersler\IE252\Case 2\IE252 Case 2 Tables.xlsx',sheet_name='Table3')
programs_data = pd.read_excel(r'C:\Users\HP\Desktop\dersler\IE252\Case 2\IE252 Case 2 Tables.xlsx',sheet_name='Table4')
programs_data = programs_data.fillna(0)

salary = {(x+1):people_data.iloc[x,1] for x in range(6)}
daily_wage_5_percent = {x+1:((salary[x+1]/250)*0.05) for x in range(6)}
required_skills = {}
program_launching_cost = {(x+1):skills_data.iloc[x,1] for x in range(15)}
provided_skills = {}
days_long = {(x+1):programs_data.iloc[x,1] for x in range(15)}
interfering_programs = {}

for m in range(6):
    for n in range(41):
        required_skills[(m+1,n+1)] = people_data.iloc[m,n+2]

for m in range(15):
    for n in range(41):
        provided_skills[(m+1,n+1)] = skills_data.iloc[m,n+2]

for m in range(15):
    for n in range(4):
        interfering_programs[(m+1,n+1)] = programs_data.iloc[m,n+2]


mdl = pyo.ConcreteModel('Case_2')

mdl.I = pyo.Set(initialize=[x+1 for x in range(6)], doc="employee")
mdl.J = pyo.Set(initialize=[x+1 for x in range(41)], doc="skills")
mdl.K = pyo.Set(initialize=[x+1 for x in range(15)], doc="programs")
mdl.M = pyo.Set(initialize=[x+1 for x in range(4)], doc="for interfering programs")

mdl.SLC = pyo.Param(mdl.I, initialize=salary, doc="salaries of each position")
mdl.DWC = pyo.Param(mdl.I, initialize=daily_wage_5_percent, doc="salaries of each position")
mdl.RS = pyo.Param(mdl.I, mdl.J, initialize=required_skills, doc="required skills for each position")
mdl.PLC = pyo.Param(mdl.K, initialize=program_launching_cost, doc="program launching costs of each program")
mdl.PS = pyo.Param(mdl.K, mdl.J, initialize=provided_skills, doc="provided skills in each program")
mdl.DL = pyo.Param(mdl.K, initialize=days_long, doc="how many days in each program last")
mdl.IPS = pyo.Param(mdl.K, mdl.M, initialize=interfering_programs, doc="which programs are interefering each other")

mdl.vX = pyo.Var(mdl.I, doc="programs total day for employee i", within=pyo.NonNegativeReals)
mdl.vY = pyo.Var(mdl.I, mdl.K, doc="if employee i is joining the program k", within=pyo.Binary)
mdl.vT = pyo.Var(mdl.I, mdl.K, mdl.J, doc="skills j that an employee i got at the end of the program", within=pyo.NonNegativeReals)
mdl.vF = pyo.Var(mdl.K, doc="if a program k is provided or not", within=pyo.NonNegativeReals)
mdl.vC = pyo.Var(mdl.K, within=pyo.Binary)


def skills_constraint(mdl, i,j,k):
    return mdl.vT[i,k,j] == np.multiply(mdl.vY[i,k] , mdl.PS[k,j])
mdl.skills_constraint = pyo.Constraint(mdl.I, mdl.J, mdl.K, rule=skills_constraint)


def skills_constraint2(mdl, i,j,k):
    return mdl.RS[i,j] <= sum(mdl.vT[i,k,j] for k in mdl.K)
mdl.skills_constraint2 = pyo.Constraint(mdl.I, mdl.J, mdl.K, rule=skills_constraint2)


def day_constraint(mdl, i,k):
    #mdl.vY[i,k]*mdl.DL[k]
    return sum(np.multiply(mdl.vY[i,k],mdl.DL[k])for k in mdl.K) == mdl.vX[i]
mdl.day_constraint = pyo.Constraint(mdl.I, mdl.K, rule=day_constraint)

def day_constraint2(mdl, i):
    #mdl.vY[i,k]*mdl.DL[k]
    return mdl.vX[i] <= 15 
mdl.day_constraint2 = pyo.Constraint(mdl.I, rule=day_constraint2)

def interfering_programs_constraint1_1(mdl, i):
   return mdl.vY[i,1] + mdl.vY[i,3] <= 1
mdl.interfering_programs_constraint1_1 = pyo.Constraint(mdl.I, rule=interfering_programs_constraint1_1)

def interfering_programs_constraint1_2(mdl, i):
   return mdl.vY[i,1] + mdl.vY[i,5] <= 1
mdl.interfering_programs_constraint1_2 = pyo.Constraint(mdl.I, rule=interfering_programs_constraint1_2)

def interfering_programs_constraint1_3(mdl, i):
   return mdl.vY[i,1] + mdl.vY[i,8] <= 1
mdl.interfering_programs_constraint1_3 = pyo.Constraint(mdl.I, rule=interfering_programs_constraint1_3)

def interfering_programs_constraint2_1(mdl, i):
   return mdl.vY[i,2] + mdl.vY[i,3] <= 1
mdl.interfering_programs_constraint2_1 = pyo.Constraint(mdl.I, rule=interfering_programs_constraint2_1)

def interfering_programs_constraint2_2(mdl, i):
   return mdl.vY[i,2] + mdl.vY[i,7] <= 1
mdl.interfering_programs_constraint2_2 = pyo.Constraint(mdl.I, rule=interfering_programs_constraint2_2)

def interfering_programs_constraint2_3(mdl, i):
   return mdl.vY[i,2] + mdl.vY[i,10] <= 1
mdl.interfering_programs_constraint2_3 = pyo.Constraint(mdl.I, rule=interfering_programs_constraint2_3)

def interfering_programs_constraint2_4(mdl, i):
   return mdl.vY[i,2] + mdl.vY[i,15] <= 1
mdl.interfering_programs_constraint2_4 = pyo.Constraint(mdl.I, rule=interfering_programs_constraint2_4)

def interfering_programs_constraint3_1(mdl, i):
   return mdl.vY[i,3] + mdl.vY[i,1] <= 1
mdl.interfering_programs_constraint3_1 = pyo.Constraint(mdl.I, rule=interfering_programs_constraint3_1)

def interfering_programs_constraint3_2(mdl, i):
   return mdl.vY[i,3] + mdl.vY[i,2] <= 1
mdl.interfering_programs_constraint3_2 = pyo.Constraint(mdl.I, rule=interfering_programs_constraint3_2)

def interfering_programs_constraint3_3(mdl, i):
   return mdl.vY[i,3] + mdl.vY[i,12] <= 1
mdl.interfering_programs_constraint3_3 = pyo.Constraint(mdl.I, rule=interfering_programs_constraint3_3)

def interfering_programs_constraint4_1(mdl, i):
   return mdl.vY[i,4] + mdl.vY[i,7] <= 1
mdl.interfering_programs_constraint4_1 = pyo.Constraint(mdl.I, rule=interfering_programs_constraint4_1)

def interfering_programs_constraint4_2(mdl, i):
   return mdl.vY[i,4] + mdl.vY[i,14] <= 1
mdl.interfering_programs_constraint4_2 = pyo.Constraint(mdl.I, rule=interfering_programs_constraint4_2)

def interfering_programs_constraint5_1(mdl, i):
   return mdl.vY[i,5] + mdl.vY[i,1] <= 1
mdl.interfering_programs_constraint5_1 = pyo.Constraint(mdl.I, rule=interfering_programs_constraint5_1)

def interfering_programs_constraint5_2(mdl, i):
   return mdl.vY[i,5] + mdl.vY[i,9] <= 1
mdl.interfering_programs_constraint5_2 = pyo.Constraint(mdl.I, rule=interfering_programs_constraint5_2)

def interfering_programs_constraint5_3(mdl, i):
   return mdl.vY[i,5] + mdl.vY[i,12] <= 1
mdl.interfering_programs_constraint5_3 = pyo.Constraint(mdl.I, rule=interfering_programs_constraint5_3)

def interfering_programs_constraint6(mdl, i):
   return mdl.vY[i,6] + mdl.vY[i,7] <= 1
mdl.interfering_programs_constraint6 = pyo.Constraint(mdl.I, rule=interfering_programs_constraint6)

def interfering_programs_constraint11(mdl, i):
   return mdl.vY[i,11] + mdl.vY[i,12] <= 1
mdl.interfering_programs_constraint11 = pyo.Constraint(mdl.I, rule=interfering_programs_constraint11)

def interfering_programs_constraint13(mdl, i):
   return mdl.vY[i,13] + mdl.vY[i,14] <= 1
mdl.interfering_programs_constraint13 = pyo.Constraint(mdl.I, rule=interfering_programs_constraint13)

def program_provided_constraint(mdl, k):
   return mdl.vF[k] == sum(mdl.vY[i,k] for i in mdl.I)
mdl.program_provided_constraint = pyo.Constraint(mdl.K, rule=program_provided_constraint)

def program_provided_constraint2(mdl,k):
    return mdl.vF[k] <= ((15)*mdl.vC[k])
mdl.program_provided_constraint2 = pyo.Constraint(mdl.K, rule=program_provided_constraint2)

def program_provided_constraint3(mdl,k):
    return mdl.vF[k] >= (1 - (15)*(1 - mdl.vC[k]))
mdl.program_provided_constraint3 = pyo.Constraint(mdl.K, rule=program_provided_constraint3)

#sum(mdl.vY[i,k] for i in mdl.I)
#sum(sum(mdl.vX[n,i] for n in mdl.N)*mdl.SLC[i] for i in mdl.I)*0.05 
def objective_function(mdl):
    a = sum(mdl.vC[k]*mdl.PLC[k] for k in mdl.K)
    b = sum(sum(mdl.vY[i,k] for k in mdl.K)*mdl.DWC[i] for i in mdl.I)
    return a+b

mdl.objective_function = pyo.Objective(rule=objective_function, sense=pyo.minimize)


Solver = SolverFactory('glpk')
SolverResults = Solver.solve(mdl)

Y_dict = mdl.vY.extract_values()
T_dict = mdl.vT.extract_values()
F_dict = mdl.vF.extract_values()
C_dict = mdl.vC.extract_values()
X_dict = mdl.vX.extract_values()

opt_z = pyo.value(mdl.objective_function)

print(opt_z)


