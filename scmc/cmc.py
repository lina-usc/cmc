import brian2 as br
import pandas as pd
from tqdm.notebook import tqdm
import numpy as np
from functools import partial
from frozendict import frozendict

##### PARAMETER VALUES #######

# https://brian2.readthedocs.io/en/stable/examples/frompapers.Stimberg_et_al_2018.example_1_COBA.html
### Neuron parameters
parameters = frozendict(
    E_e = 0*br.mV,             # Excitatory synaptic reversal potential
    E_i = -80*br.mV,           # Inhibitory synaptic reversal potential
    V_th = -50*br.mV,          # Firing threshold (just used for v initialization)
    E_l = -60*br.mV,           # Leak reversal potential (just used for v initialization)

    C_m = 198*br.pF,           # Membrane capacitance

    tau_e = 5*br.ms,           # Excitatory synaptic time constant
    tau_i = 10*br.ms,          # Inhibitory synaptic time constant
    tau_r = 5*br.ms,           # Refractory period

    ### Synapse parameters
    w_e = 25*br.nS,          # Excitatory synaptic conductance
    w_i = 10*br.nS,           # Inhibitory synaptic conductance
    #Omega_d = 2.0/second   # Synaptic depression rate
    #Omega_f = 3.33/second  # Synaptic facilitation rate
)


# https://groups.google.com/g/briansupport/c/39rcKe5xdsE
def get_izhikevich_pop(a, b, c, d, I=None, nb_neurons=1):
    ''' Izhikevich model '''

    if I is None:
        I = 15*br.mV/br.ms

    #Neuronal equations of the Izhikevich-neuron
    model = '''
    I_syn_e = g_e*(E_e-v)                                      : ampere 
    I_syn_i = g_i*(E_i-v)                                      : ampere 
    v_syn = (I_syn_e + I_syn_i)/C_m                            : volt/second
    dv/dt = (0.04/ms/mV)*v**2 + (5/ms)*v + 140*mV/ms - u + I + v_syn : volt #(unless refractory)
    du/dt = a*(b*v-u)                                        : volt/second
    a                                                        : 1/second
    b                                                        : 1/second
    c                                                        : volt
    d                                                        : volt/second
    I                                                        : volt/second
    dg_e/dt = -g_e/tau_e  : siemens  # post-synaptic exc. conductance
    dg_i/dt = -g_i/tau_i  : siemens  # post-synaptic inh. conductance
    '''

    reset = '''
    v = c
    u = u + d
    '''

    #Set up neuron population
    G = br.NeuronGroup(nb_neurons, model, threshold='v >= 30*mV',
                       reset=reset, #refractory="tau_r",
                       method='euler',
                       namespace=dict(parameters))

    # Creating some level of randomness in the exact parameter values, to
    # avoid all neurons ending up having the same activity after the 
    # initialization transients are gone (95% CI approximately within 
    # +/- 1% of relative error)
    G.a = a*(1 + np.random.randn()/1.96*0.01)
    G.b = b*(1 + np.random.randn()/1.96*0.01)
    G.c = c*(1 + np.random.randn()/1.96*0.01)
    G.d = d*(1 + np.random.randn()/1.96*0.01)
    G.v = c
    G.u = b * c
    G.I = I
    G.v = 'E_l + rand()*(V_th-E_l)'
    G.g_e = 'rand()*w_e'
    G.g_i = 'rand()*w_i'

    return G


# From https://brian2.readthedocs.io/en/stable/resources/tutorials/2-intro-to-brian-synapses.html
def get_stdp_synapses(pop_pre, pop_post, prob, kind="e",
                      tau_pre=None, tau_post=None, w_max=None, A_pre=None):
  
  if w_max is None:
    w_max = 0.01*br.nS
  if A_pre is None:
    A_pre = w_max
  if tau_pre is None:
    tau_pre = 20*br.ms
  if tau_post is None:
    tau_post = 20*br.ms

  s = br.Synapses(pop_pre, pop_post,
              '''
              w : siemens
              A_pre_ : siemens
              tau_pre_ : second
              tau_post_ : second
              w_max : siemens
              A_post_ : siemens
              dapre/dt = -apre/tau_pre_ : siemens (event-driven)
              dapost/dt = -apost/tau_post_ : siemens (event-driven)
              ''',
              on_pre=f'''
              g_{kind}_post += w
              apre += A_pre_
              w = clip(w+apost, 0*nS, w_max)
              ''',
              on_post='''
              apost += A_post_
              w = clip(w+apre, 0*nS, w_max)
              ''')
  
  s.connect(p=prob)  
  
  s.A_pre_ = A_pre
  s.tau_pre_ = tau_pre
  s.tau_post_ = tau_post
  s.w_max = w_max
  s.A_post_ = -A_pre*tau_pre/tau_post*1.05

  return s


class CMC:
    
    pop_labels = ("L13_exc", "L13_inh", "L4_exc",
                  "L4_inh", "L56_exc", "L56_inh")

    def __init__(self, pop_params=None, con_factor=0.25, input_kwargs=None):
        ''' '''

        if input_kwargs is None:
           input_kwargs = {}
        self.create_populations(pop_params)
        self.set_within_cmc_con(con_factor)
        self.set_inputs(**input_kwargs)
        self.add_synapses(con_factor)
        self.add_monitors()
        self.create_network()

    def set_inputs(self, n_inputs=20, input_fct=None):
        
        if input_fct is None:
            def input_fct(n_inputs):
                return br.PoissonGroup(n_inputs, rates=np.linspace(10, 110, n_inputs)*br.Hz)

        # Creating Poisson inputs
        self.pops["feedback_input_L56"] = input_fct(n_inputs)
        self.pops["feedback_input_L13"] = input_fct(n_inputs)
        self.pops["feedforward_input_L4"] = input_fct(n_inputs)      


    def run_simulation(self, stim_desc, sim_dur):
        _, rows = zip(*list(stim_desc.iterrows()))
        for row0, row1 in zip(rows[:-1], rows[1:]):
            self.pops[row0["pop"]].active = row0["status"]
            dt = row1["time"] -  row0["time"]
            if dt > 0:
                self.network.run(dt*br.second, report="text")

        self.pops[row1["pop"]].active = row1["status"]
        dt = sim_dur -  row0["time"]
        if dt > 0:
            self.network.run(dt*br.second, report="text")

    @staticmethod
    def get_default_pop_params(**params):
        ret_params = dict(a=0.02/br.ms, b=0.2/br.ms, c=-65*br.mV,
                          d=8*br.mV/br.ms, I=0*br.mV/br.ms, nb_neurons=50)
        ret_params.update(params)
        return ret_params

    def create_populations(self, pop_params=None):
        
        self.layers = ["L13", "L4", "L56"]

        if pop_params is None:
            params = {label: self.get_default_pop_params()
                      for label in self.pop_labels}
        else:
            params = {label: self.get_default_pop_params(**pop_params[label])
                      for label in pop_params}

        self.pops = {label: get_izhikevich_pop(**params[label])
                     for label in params}
        
    def set_within_cmc_con(self, con_factor):

        self.within_cmc_con = pd.DataFrame(np.zeros((6, 6))*np.nan, columns=self.pops.keys(), 
                                           index=pd.Series(self.pops.keys(), name="pre"))

        # [pre, post]
        for layer in self.layers:
          self.within_cmc_con.loc[f"{layer}_exc", f"{layer}_inh"] = 0
          self.within_cmc_con.loc[f"{layer}_inh", f"{layer}_exc"] = 0

        self.within_cmc_con.loc["L13_exc", "L56_exc"] = 1
        self.within_cmc_con.loc["L4_exc", "L56_exc"] = 1
        self.within_cmc_con.loc["L56_exc", "L4_inh"] = 1
        self.within_cmc_con.loc["L56_exc", "L4_exc"] = 2

        self.within_cmc_con = con_factor**self.within_cmc_con
        self.within_cmc_con = self.within_cmc_con.melt(ignore_index=False, var_name="post")\
                                                 .dropna().set_index("post", append=True)["value"]

    def add_synapses(self, con_factor):
        self.syns = {}
        for (pre, post), prob in tqdm(list(self.within_cmc_con.items())):
            print(pre, post, prob)

            self.syns[(pre, post)] = get_stdp_synapses(self.pops[pre], self.pops[post], prob, 
                                                  w_max=w_e if 'exc' in pre else w_i, 
                                                  kind="e" if 'exc' in pre else "i")

        amp_mult = {"mod": 0.2, "driver": 1.0}
        exponent = {("feedback", "56", "inh", "mod"): 3, 
                    ("feedback", "56", "inh", "driver"): 2, 
                    ("feedback", "56", "exc", "mod"): 4, 
                    ("feedback", "56", "exc", "driver"): 3,
                    ("feedback", "13", "inh", "mod"): 3, 
                    ("feedback", "13", "inh", "driver"): 2, 
                    ("feedback", "13", "exc", "mod"): 4, 
                    ("feedback", "13", "exc", "driver"): 3,
                    ("feedforward", "4", "exc", "driver"): 2}
        for key in exponent:
            direction, layer, pop, kind = key
            print("inputs", direction, layer, pop, kind)
            syn_key = (f"{direction}_input_{kind}", f"L{layer}_{pop}")
            self.syns[syn_key] = br.Synapses(self.pops[f"{direction}_input_L{layer}"], 
                                             self.pops[f"L{layer}_{pop}"], 
                                             on_pre=f'g_e+={amp_mult[kind]}*w_e')                                                    
            self.syns[syn_key].connect(p=con_factor**exponent[key])

    def add_monitors(self):
        self.monitors = {}
        self.spike_monitors = {}
        for pop_name, pop in self.pops.items():
          self.spike_monitors[pop_name] = br.SpikeMonitor(pop)
          if "input" in pop_name:
            continue
          self.monitors[pop_name] = br.StateMonitor(pop, ['v', 'g_e', 'g_i'], record=True)  
        
    def create_network(self):
        self.network = br.Network()
        for collection in [self.pops, self.syns, self.monitors, self.spike_monitors]:
          for item in collection.values():
            self.network.add(item)
