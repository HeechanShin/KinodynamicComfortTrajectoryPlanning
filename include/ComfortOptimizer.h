//
// Created by heechanshin on 18. 9. 26.
//

#ifndef COMFORT_TRAJECTORY_OPTIMIZER_COMFORTOPTIMIZER_H
#define COMFORT_TRAJECTORY_OPTIMIZER_COMFORTOPTIMIZER_H

#include <coin/IpTNLP.hpp>
#include <coin/IpIpoptApplication.hpp>
#include <coin/IpSolveStatistics.hpp>

#include "RobotModel.h"

using namespace std;
using namespace Ipopt;

template <const int stateDim, const int actionDim>
class ComfortOptimizer : public TNLP{
public:
    ComfortOptimizer(RobotModel<stateDim, actionDim>* robot_){
        robot = robot_;
    };
    virtual ~ComfortOptimizer(){};

    void plan(Eigen::Matrix<double, stateDim+actionDim, 1> startState_, Eigen::Matrix<double, stateDim+actionDim, 1> goalState_){
        robot->plan(startState_, goalState_);
    }

    virtual bool get_nlp_info(int& n, int& m, int& nnz_jac_g,
                              int& nnz_h_lag, IndexStyleEnum& int_style){
        return robot->getNLPInfo(n, m, nnz_jac_g, nnz_h_lag, int_style);
    }

    virtual bool get_bounds_info(int n, double* x_l, double* x_u,
                                 int m, double* g_l, double* g_u){
        return robot->getBoundInfo(n, x_l, x_u, m, g_l, g_u);
    }

    virtual bool get_starting_point(int n, bool init_x, double* x,
                                    bool init_z, double* z_L, double* z_U,
                                    int m, bool init_lambda,
                                    double* lambda){
        return robot->getStartingPoint(n, init_x, x, init_z, z_L, z_U, m, init_lambda, lambda);
    }

    virtual bool eval_f(int n, const double* x, bool new_x, double& obj_value){
        return robot->getObjective(n, x, new_x, obj_value);
    }

    virtual bool eval_grad_f(int n, const double* x, bool new_x, double* grad_f){
        return robot->getGradient(n, x, new_x, grad_f);
    }

    virtual bool eval_g(int n, const double* x, bool new_x, int m, double* g){
        return robot->getConstraints(n, x, new_x, m, g);
    }

    virtual bool eval_jac_g(int n, const double* x, bool new_x,
                            int m, int nele_jac, int* iRow, int *jCol,
                            double* values){
        return robot->getJacobian(n, x, new_x, m, nele_jac, iRow, jCol, values);
    }

    virtual bool eval_h(int n, const double* x, bool new_x,
                        double obj_factor, int m, const double* lambda,
                        bool new_lambda, int nele_hess, int* iRow,
                        int* jCol, double* values){
        return robot->getHessian(n, x, new_x, obj_factor, m, lambda, new_lambda, nele_hess, iRow, jCol, values);
    }

    virtual void finalize_solution(SolverReturn status,
                                   int n, const double* x, const double* z_L, const double* z_U,
                                   int m, const double* g, const double* lambda,
                                   double obj_value,
                                   const IpoptData* ip_data,
                                   IpoptCalculatedQuantities* ip_cq){
        robot->finalizeSolution(n, x);
    }

private:
    RobotModel<stateDim, actionDim>* robot;
};

#endif //COMFORT_TRAJECTORY_OPTIMIZER_COMFORTOPTIMIZER_H
