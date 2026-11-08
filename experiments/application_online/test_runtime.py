"""Numerical checks for application import, warm starts and old-hierarchy reuse."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='1'
import unittest
import numpy as np
from scipy.sparse import diags
from hypre.bindings import create_env, augment_setup_params
from hypre.bindings.config import configure_smoother_profile
from setup.space import DEFAULT_SETUP_PARAMS
from experiments.application_online.run import Pcg, true_residual


class ApplicationRuntimeTests(unittest.TestCase):
    def setUp(self):
        configure_smoother_profile('l1_jacobi_direct_coarse')
        self.A=diags([-np.ones(199),2.1*np.ones(200),-np.ones(199)],[-1,0,1]).tocsr()
        self.truth=np.random.default_rng(19).normal(size=200)
        self.b=self.A@self.truth

    def test_full_and_cycle_solve_agree_on_warm_start_accuracy(self):
        x0=.9*self.truth
        solutions=[]
        for cycle_mode in (False,True):
            out=np.empty(200)
            with create_env(matrix=self.A,rhs=self.b,initial_guess=x0,solution_out=out) as env:
                if cycle_mode:
                    prep=env.prepare_rl(augment_setup_params(DEFAULT_SETUP_PARAMS))
                    self.assertAlmostEqual(prep.initial_residual_norm,.1,places=12)
                    for _ in range(100):
                        residual,_=env.step_rl(relax_weight=1.,sweeps_down=1,sweeps_up=1,
                                               tol=1e-9,max_cycles=100)
                        if residual<1e-9:break
                else:
                    env.solve(params=augment_setup_params(DEFAULT_SETUP_PARAMS),tol=1e-9,max_iter=100)
            self.assertLess(true_residual(self.A,self.b,out),1e-9)
            solutions.append(out)
        self.assertLess(np.linalg.norm(solutions[0]-solutions[1]),1e-6)

    def test_reused_hierarchy_solves_changed_operator(self):
        solver=Pcg(True)
        try:
            for i in range(3):
                A=self.A+diags(np.linspace(.2,2.,200)*i,format='csr')
                b=A@self.truth
                x,stats=solver.solve(A,b,np.zeros(200),tol=1e-9)
                self.assertTrue(stats['converged'])
                self.assertEqual(stats['builds'],int(i==0))
                self.assertLess(np.linalg.norm(x-self.truth),1e-6)
        finally:solver.close()

    def test_exact_warm_start_skips_build(self):
        solver=Pcg()
        x,stats=solver.solve(self.A,self.b,self.truth)
        self.assertEqual(stats['iterations'],0)
        self.assertEqual(stats['builds'],0)
        self.assertIsNone(solver.env)

    def test_generated_problem_keeps_original_normalization(self):
        with create_env(nx=4,ny=4,nz=4,stencil=0) as env:
            prep=env.prepare_rl(augment_setup_params(DEFAULT_SETUP_PARAMS))
            self.assertEqual(prep.initial_residual_norm,1.)


if __name__=='__main__':unittest.main()
