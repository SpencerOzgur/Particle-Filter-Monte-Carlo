import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

def log_n(x,m,s):
    
        return (
            -0.5*np.log(2*np.pi)
                - np.log(s)
        - 0.5*((x-m)/s)**2
        )


def obj_particle_filter(params, gdp_obs):

    phi, theta, sig, tau = params

    
    rng = np.random.default_rng(123)

    
    n_particles = 1000
    T = len(gdp_obs)

    particles = rng.normal(0, np.std(gdp_obs), n_particles)
    
    ll = 0
    
    for t in range(1, T):

        gap_last = particles.copy()

        #Transition density
        m_T = phi * gap_last
    
        #Proposal dist
        sig_I = np.sqrt( 
            1 / ((1/sig**2) + (theta**2 / tau**2))
        )
            
        m_I = (
            sig_I**2
            * ((m_T / sig**2)
               + (theta * gdp_obs[t] / tau**2)
              )
        )


        gap_t = m_I + sig_I*rng.standard_normal(n_particles)
        gap_new = gap_t
        

        #Likelihood function
        m_L = theta * gap_new

        log_p_gdp = log_n(gdp_obs[t], m_L, tau)

        log_p_gap = log_n(gap_t, m_T, sig)

        #Proposal density
        log_q = log_n(gap_t, m_I, sig_I)

        #Set weight
        log_w = (
            -np.log(n_particles)
            + log_p_gdp
            + log_p_gap
            - log_q
        )
    
        #Add to likelihood
        addition = logsumexp(log_w)

        if not np.isfinite(addition):
            return 1e12
        
        ll += addition
    
        #Normalize weights
        w_normalized = np.exp(log_w - addition)
    
        ###Sampling Importance Resampling
        c = np.cumsum(w_normalized)
        #Enforce total cumulative sum is 1 in case of numeric error
        c[-1] = 1

        i = 0
        u_sample = rng.uniform()
        
        for j in range(n_particles):
            u = (1/n_particles) * (u_sample + j)
            
            while u > c[i]:
                i += 1
                
            particles[j] = gap_new[i]
            
    return -ll



def get_pf_params(gdp_obs):

    #Initial guesses

    phi_0 = 0.5
    theta_0 = 0.5

    sig_0 = np.std(gdp_obs)
    tau_0 = np.std(gdp_obs)
    
    
    x0 = np.array([phi_0, theta_0, 
                   sig_0, tau_0])


    #Parameter bounds

    phi_lb = -0.999
    phi_ub = 0.999

    theta_lb = 1e-4
    theta_ub = 1e4

    sig_lb = 1e-4
    sig_ub = 1e5

    tau_lb = 1e-4
    tau_ub = 1e5
    
    
    lb_list = [phi_lb, theta_lb, sig_lb, tau_lb]
    ub_list = [phi_ub, theta_ub, sig_ub, tau_ub]

    bounds = list(zip(lb_list, ub_list))

    sol = minimize(
        lambda params: obj_particle_filter(params, gdp_obs),
        bounds=bounds, x0=x0, method='Nelder-Mead'
    )

    return sol.x

def recover_gap_pf(optimal_params, gdp_obs):

    phi, theta, sig, tau = optimal_params

    rng = np.random.default_rng(123)
    
    n_particles = 1000
    T = len(gdp_obs)

    particles = rng.normal(0, np.std(gdp_obs), n_particles)

    #Store estimates
    gap_estimates = np.zeros(T)
    gap_var_estimates = np.zeros(T)

    gap_estimates[0] = np.mean(particles)
    gap_var_estimates[0] = np.var(particles)
    
    for t in range(1, T):

        gap_last = particles.copy()

        #Transition density
        m_T = phi * gap_last
    
        #Proposal dist
        sig_I = np.sqrt( 
            1 / ((1/sig**2) + (theta**2 / tau**2))
        )
            
        m_I = (
            sig_I**2
            * ((m_T / sig**2)
               + (theta * gdp_obs[t] / tau**2)
              )
        )


        gap_t = m_I + sig_I*rng.standard_normal(n_particles)
        gap_new = gap_t
        

        #Likelihood function
        m_L = theta * gap_new

        log_p_gdp = log_n(gdp_obs[t], m_L, tau)

        log_p_gap = log_n(gap_t, m_T, sig)

        #Proposal density
        log_q = log_n(gap_t, m_I, sig_I)

        #Set weight
        log_w = (
            -np.log(n_particles)
            + log_p_gdp
            + log_p_gap
            - log_q
        )

        addition = logsumexp(log_w)
    

        #Normalize weights
        w_normalized = np.exp(log_w - addition)


        #Store values
        gap_estimates[t] = np.sum(
            w_normalized * gap_new
        )
        
        gap_var_estimates[t] = np.sum(
            w_normalized * (gap_new - gap_estimates[t])**2
        )
            

        
    
        ###Sampling Importance Resampling
        c = np.cumsum(w_normalized)
        #Enforce total cumulative sum is 1 in case of numeric error
        c[-1] = 1

        i = 0
        u_sample = rng.uniform()
        
        for j in range(n_particles):
            u = (1/n_particles) * (u_sample + j)
            
            while u > c[i]:
                i += 1
                
            particles[j] = gap_new[i]
            
    return gap_estimates, gap_var_estimates
