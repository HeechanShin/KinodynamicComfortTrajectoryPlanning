//
// Created by heechanshin on 18. 9. 26.
//

#ifndef COMFORT_TRAJECTORY_OPTIMIZER_ROBOTMODEL_H
#define COMFORT_TRAJECTORY_OPTIMIZER_ROBOTMODEL_H

#include <cmath>

#include <ros/ros.h>
#include <bod_srvs/Bod.h>
#include <std_msgs/Float64MultiArray.h>

#include <coin/IpTNLP.hpp>
#include <coin/IpIpoptApplication.hpp>

#include <eigen3/Eigen/Dense>

using namespace std;
using namespace Ipopt;
using namespace Eigen;

template <const int stateDim, const int actionDim>
class RobotModel{
public:
    RobotModel(int N_, ros::ServiceClient& client_, ros::Publisher& pub_, ros::Publisher& obs_){
        int argc;
        char** argv;
        N = N_;
        client = client_;
        pubTrajectory = pub_;
        pubDetectedObstacle = obs_;

        bod_srvs::Bod bod;
        if(client.call(bod)) {
            std::vector<int> map = bod.response.map;
            resolution = bod.response.resolution;
            int mapWidthInGrid = (int) bod.response.map_width;
            int mapHeightInGrid = (int) bod.response.map_height;
            mapWidth = mapWidthInGrid * resolution;
            double mapHeight = mapHeightInGrid * resolution;

            gridMap = Map<MatrixXi>(map.data(), mapHeightInGrid, mapWidthInGrid);
            gridMap.colwise().reverseInPlace();
            gridMap.transposeInPlace();
        }
        TRACING_LIMIT_IN = mapWidth * sqrt(2.0);
        TRACING_LIMIT_OUT = l * 8.0;
        EPSILON = l * 3.0;
        hermite_S = MatrixXd::Zero(4, N+1);
        VectorXd s = VectorXd::LinSpaced(N+1, 0.0, 1.0);
        for(int i = 0; i < N+1; i++){
            hermite_S(0, i) = pow(s(i), 3);
            hermite_S(1, i) = pow(s(i), 2);
            hermite_S(2, i) = pow(s(i), 1);
            hermite_S(3, i) = 1.0;
        }
        hermite_h = MatrixXd::Zero(4, 4);
        hermite_h << 2, -2, 1, 1, -3, 3, -2, -1, 0, 0, 1, 0, 1, 0, 0, 0;
        final_x_left = new double[N+1]();
        final_y_left = new double[N+1]();
        final_x_right = new double[N+1]();
        final_y_right = new double[N+1]();
        obs_grad_x = new double[N+1]();
        obs_grad_y = new double[N+1]();
        obs_hess_xx = new double[N+1]();
        obs_hess_yy = new double[N+1]();
        obs_hess_xy = new double[N+1]();
        dist = new double[N+1]();
        dist_grad_x = new double[N+1]();
        dist_grad_y = new double[N+1]();
        dist_hess_xx = new double[N+1]();
        dist_hess_yy = new double[N+1]();
        dist_hess_xy = new double[N+1]();
        comfort_grad_tf = new double[N]();
        comfort_grad_v = new double[N]();
        comfort_grad_phi = new double[N]();
        comfort_grad_a = new double[N]();
        comfort_grad_v_n = new double[N]();
        comfort_grad_phi_n = new double[N]();
        comfort_grad_a_n = new double[N]();
        comfort_hess_tfv = new double[N]();
        comfort_hess_tfphi = new double[N]();
        comfort_hess_tfa = new double[N]();
        comfort_hess_vv = new double[N]();
        comfort_hess_vphi = new double[N]();
        comfort_hess_phiphi = new double[N]();
        comfort_hess_aa = new double[N]();
        comfort_hess_tfv_n = new double[N]();
        comfort_hess_tfphi_n = new double[N]();
        comfort_hess_tfa_n = new double[N]();
        comfort_hess_vv_n = new double[N]();
        comfort_hess_vphi_n = new double[N]();
        comfort_hess_phiphi_n = new double[N]();
        comfort_hess_aa_n = new double[N]();
    }
    ~RobotModel(){
        delete (final_x_left);
        delete (final_y_left);
        delete (final_x_right);
        delete (final_y_right);
        delete (obs_grad_x);
        delete (obs_grad_y);
        delete (obs_hess_xx);
        delete (obs_hess_yy);
        delete (obs_hess_xy);
        delete (dist);
        delete (dist_grad_x);
        delete (dist_grad_y);
        delete (dist_hess_xx);
        delete (dist_hess_yy);
        delete (dist_hess_xy);
        delete (comfort_grad_tf);
        delete (comfort_grad_v);
        delete (comfort_grad_phi);
        delete (comfort_grad_a);
        delete (comfort_grad_v_n);
        delete (comfort_grad_phi_n);
        delete (comfort_grad_a_n);
        delete (comfort_hess_tfv);
        delete (comfort_hess_tfphi);
        delete (comfort_hess_tfa);
        delete (comfort_hess_vv);
        delete (comfort_hess_vphi);
        delete (comfort_hess_phiphi);
        delete (comfort_hess_aa);
        delete (comfort_hess_tfv_n);
        delete (comfort_hess_tfphi_n);
        delete (comfort_hess_tfa_n);
        delete (comfort_hess_vv_n);
        delete (comfort_hess_vphi_n);
        delete (comfort_hess_phiphi_n);
        delete (comfort_hess_aa_n);
    }

    int getVariablesSize(){
        return 1 + (stateDim + actionDim)*(N+1);
    }

    int getConstraintsSize(){
        return stateDim*(N + 2) + N+1;
    }

    int getJacobianSize(){
        return 31*N + 10 + 3*(N+1);
    }

    int getHessianSize(){
        return 14*(N + 1);
    }

    void plan(Eigen::Matrix<double, stateDim+actionDim, 1> startState_, Eigen::Matrix<double, stateDim+actionDim, 1> goalState_){
        startState = startState_;
        goalState = goalState_;
    }

    bool getNLPInfo(int& n, int& m, int& nnz_jac_g, int& nnz_h_lag, Ipopt::TNLP::IndexStyleEnum& index_style){
        n = getVariablesSize();
        m = getConstraintsSize();
        nnz_jac_g = getJacobianSize();
        nnz_h_lag = getHessianSize();

        index_style = Ipopt::TNLP::C_STYLE;

        return true;
    }

    bool getBoundInfo(int n, double* x_l, double* x_u, int m, double* g_l, double* g_u){
        int c = 0;
        x_l[c++] = 0.0;
        for(int k = 0; k < N+1; k++){
            x_l[c++] = -mapWidth/2.0;
        }
        for(int k = 0; k < N+1; k++){
            x_l[c++] = -mapWidth/2.0;
        }
        for(int k = 0; k < N+1; k++){
            x_l[c++] = -2.0;
        }
        for(int k = 0; k < N+1; k++){
            x_l[c++] = -M_PI;
        }
        for(int k = 0; k < N+1; k++){
            x_l[c++] = -M_PI_4;
        }
        for(int k = 0; k < N+1; k++){
            x_l[c++] = -10.0;
        }
        for(int k = 0; k < N+1; k++){
            x_l[c++] = -M_PI_2;
        }

        int d = 0;
        x_u[d++] = INFINITY;
        for(int k = 0; k < N+1; k++){
            x_u[d++] = mapWidth/2.0;
        }
        for(int k = 0; k < N+1; k++){
            x_u[d++] = mapWidth/2.0;
        }
        for(int k = 0; k < N+1; k++){
            x_u[d++] = 2.0;
        }
        for(int k = 0; k < N+1; k++){
            x_u[d++] = M_PI;
        }
        for(int k = 0; k < N+1; k++){
            x_u[d++] = M_PI_4;
        }
        for(int k = 0; k < N+1; k++){
            x_u[d++] = 10.0;
        }
        for(int k = 0; k < N+1; k++){
            x_u[d++] = M_PI_2;
        }

        for(int i = 0; i < m; i++){
            g_l[i] = 0;
            g_u[i] = 0;
        }

        for(int i = 0; i < N+1; i++){
            g_l[(m-1) - i] = -INFINITY;
            g_u[(m-1) - i] = Cmax;
        }

        assert((c == n) && (d == n));
        return (c == n) && (d == n);
    }

    bool getStartingPoint(int n, bool init_x, Number* x, bool init_z, double* z_L, double* z_U, int m, bool init_lambda, double* lambda){
        int c = 0;
        double tf = 10.0;
        double h = tf/N;

        Eigen::Matrix<double, 4, 2> hermite_C;
        hermite_C << startState(0, 0), startState(1, 0), goalState(0, 0), goalState(1, 0), hermite_coeff*(abs(startState(2, 0)) + 0.0001)*cos(startState(3, 0)), hermite_coeff*(abs(startState(2, 0)) + 0.0001)*sin(startState(3, 0)), hermite_coeff*(abs(startState(2, 0)) + 0.0001)*cos(goalState(3, 0)), hermite_coeff*(abs(startState(2, 0)) + 0.0001)*sin(goalState(3, 0));

        Eigen::MatrixXd hermite_p = hermite_S.transpose() * hermite_h *hermite_C;

        // VectorXd xs = VectorXd::LinSpaced(N + 1, startState(0, 0), goalState(0, 0));
        VectorXd xs(N + 1);
        for(int i = 0; i < N + 1; i ++){
            xs(i, 0) = hermite_p(i, 0);
        }
        // VectorXd ys = VectorXd::LinSpaced(N + 1, startState(1, 0), goalState(1, 0));
        VectorXd ys(N + 1);
        for(int i = 0; i < N + 1; i++){
            ys(i, 0) = hermite_p(i, 1);
        }
        VectorXd vs = VectorXd::LinSpaced(N + 1, startState(2, 0), goalState(2, 0));
        // VectorXd ths = LinSpacedAngle(N + 1, startState(3, 0), goalState(3, 0));
        VectorXd ths(N + 1);
        ths(0, 0) = startState(3, 0);
        for(int i = 1; i < N; i++){
            ths(i, 0) = atan2(ys(i+1, 0) - ys(i, 0), xs(i+1, 0) - xs(i, 0));
        }
        ths(N, 0) = goalState(3, 0);
        // VectorXd phis = LinSpacedAngle(N + 1, startState(4, 0), goalState(4, 0));
        VectorXd phis(N + 1);
        phis(0, 0) = startState(4, 0);
        for(int i = 1; i < N; i++){
            phis(i, 0) = atan2((ths(i+1, 0) - ths(i, 0))*l, h*vs(i, 0));
        }
        phis(N, 0) = goalState(4, 0);
        VectorXd as = VectorXd::LinSpaced(N + 1, startState(5, 0), goalState(5, 0));
        VectorXd ws = VectorXd::LinSpaced(N + 1, startState(6, 0), goalState(6, 0));

        x[0] = tf;
        for(int i = 0; i < N+1; i++){
            x[1 + i] = xs(i, 0);
            x[1 + (N+1) + i] = ys(i, 0);
            x[1 + 2*(N+1) + i] = vs(i, 0);
            x[1 + 3*(N+1) + i] = ths(i, 0);
            x[1 + 4*(N+1) + i] = phis(i, 0);
            x[1 + 5*(N+1) + i] = as(i, 0);
            x[1 + 6*(N+1) + i] = ws(i, 0);
        }

        std_msgs::Float64MultiArray msg;
        msg.layout.data_offset = N;
        for(int i = 0; i < getVariablesSize(); i++){
            msg.data.push_back(x[i]);
        }
        pubTrajectory.publish(msg);

        return true;
    }

    bool getObjective(int n, const double* x, bool new_x, double& obj_value){
        double h = x[0]/N;

        double objTf_;
        double objComfort_;
        double objObs_;

        objTf_ = tf_coeff > 0 ? getTfObjective(x, tf_coeff) : 0.0;
        objComfort_ = comfort_coeff > 0 ? getComfortObjective(x, comfort_coeff) : 0.0;
        objObs_ = obs_coeff > 0 ? getObsObjective(x, obs_coeff) : 0.0;

        obj_value = objTf_ + objComfort_ + objObs_;

        return true;
    }

    inline double getTfObjective(const double* x, double coeff){
        return coeff * x[0];
    }

    inline double getComfortObjective(const double* x, double coeff){
        double objComfort = 0.0;

        double h = x[0]/N;
        for(int k = 0; k < N; k++){
            double v = x[1 + 2*(N+1) + k];
            double v_n = x[1 + 2*(N+1) + k+1];
            double phi = x[1 + 4*(N+1) + k];
            double phi_n = x[1 + 4*(N+1) + k+1];
            double a = x[1 + 5*(N+1) + k];
            double a_n = x[1 + 5*(N+1) + k+1];

            objComfort += (h/2.0)*(pow(a,2) + pow(tan(phi)/l,2)*pow(v,4)+pow(a_n,2) + pow(tan(phi_n)/l,2)*pow(v_n,4));

            comfort_grad_tf[k] = coeff * (pow(a,2) + pow(a_n,2) + pow((pow(v,2)*tan(phi))/l,2) + pow((pow(v_n,2)*tan(phi_n))/l,2))/(2.0*N);
            comfort_grad_v[k] = coeff * h*2.0*pow(tan(phi), 2)*pow(v,3)/pow(l,2);
            comfort_grad_phi[k] = coeff * h*pow(v,4)*tan(phi)*(pow(tan(phi),2)+1)/pow(l,2);
            comfort_grad_a[k] = coeff * h*a;
            comfort_grad_v_n[k] = coeff * h*2.0*pow(tan(phi_n), 2)*pow(v_n,3)/pow(l,2);
            comfort_grad_phi_n[k] = coeff * h*pow(v_n,4)*tan(phi_n)*(pow(tan(phi_n),2)+1)/pow(l,2);
            comfort_grad_a_n[k] = coeff * h*a_n;
            comfort_hess_tfv[k] = coeff * 2.0/N*pow(v,3)*pow(tan(phi),2)/pow(l,2);
            comfort_hess_tfphi[k] = coeff * pow(v,4)*tan(phi)*(pow(tan(phi),2)+1)/(N*pow(l,2));
            comfort_hess_tfa[k] = coeff * a/N;
            comfort_hess_vv[k] = coeff * 6.0*h*pow(v*tan(phi)/l,2);
            comfort_hess_vphi[k] = coeff * 4.0*h*pow(v,3)*tan(phi)*(pow(tan(phi),2)+1)/pow(l,2);
            comfort_hess_phiphi[k] = coeff * (h*pow(pow(v,2)*(pow(tan(phi),2)+1)/l,2) + 2.0*h*pow(v,4)*pow(tan(phi),2)*(pow(tan(phi),2)+1)/pow(l,2));
            comfort_hess_aa[k] = coeff * h;
            comfort_hess_tfv_n[k] = coeff * 2.0/N*pow(v_n,3)*pow(tan(phi_n),2)/pow(l,2);
            comfort_hess_tfphi_n[k] = coeff * pow(v_n,4)*tan(phi_n)*(pow(tan(phi_n),2)+1)/(N*pow(l,2));
            comfort_hess_tfa_n[k] = coeff * a_n/N;
            comfort_hess_vv_n[k] = coeff * 6.0*h*pow(v_n*tan(phi_n)/l,2);
            comfort_hess_vphi_n[k] = coeff * 4.0*h*pow(v_n,3)*tan(phi_n)*(pow(tan(phi_n),2)+1)/pow(l,2);
            comfort_hess_phiphi_n[k] = coeff * (h*pow(pow(v_n,2)*(pow(tan(phi_n),2)+1)/l,2) + 2.0*h*pow(v_n,4)*pow(tan(phi_n),2)*(pow(tan(phi_n),2)+1)/pow(l,2));
            comfort_hess_aa_n[k] = coeff * h;
        }

        return coeff * objComfort;
    }

    inline double getObsObjective(const double* x, double coeff){
        getDistanceObjective(x, coeff);

        double obsObj = 0.0;
        for(int i = 0; i < N+1; i++){
            obsObj += coeff * (c(dist[i]) / N);
            obs_grad_x[i] = coeff * (dc(dist[i], dist_grad_x[i]) / N);
            obs_grad_y[i] = coeff * (dc(dist[i], dist_grad_y[i]) / N);
            obs_hess_xx[i] = coeff * (ddc(dist[i], dist_grad_x[i], dist_grad_x[i], dist_hess_xx[i]) / N);
            obs_hess_yy[i] = coeff * (ddc(dist[i], dist_grad_y[i], dist_grad_y[i], dist_hess_yy[i]) / N);
            obs_hess_xy[i] = coeff * (ddc(dist[i], dist_grad_x[i], dist_grad_y[i], dist_hess_xy[i]) / N);
        }
        return obsObj;
    }

    void getDistanceObjective(const double* x, const double coeff){
        for(int k = 0; k < N+1; k++){
            double x_cord = x[1 + k];
            double y_cord = x[1 + N+1 + k];
            double th_cord = x[1 + 3*(N+1) + k];

            int x_idx;
            int y_idx;
            getGridIndex(x_cord, y_cord, x_idx, y_idx, mapWidth, resolution);

            if(gridMap(y_idx, x_idx) != 0){
                // inside
                // DDA
                /*double angle_l = th_cord + M_PI_2;

                int dest_x_idx_left;
                int dest_y_idx_left;
                getGridIndex(x_cord + TRACING_LIMIT_IN*cos(angle_l), y_cord + TRACING_LIMIT_IN*sin(angle_l), dest_x_idx_left, dest_y_idx_left, mapWidth, resolution);
                int dest_x_idx_right;
                int dest_y_idx_right;
                getGridIndex(x_cord + TRACING_LIMIT_IN*cos(angle_l-M_PI), y_cord + TRACING_LIMIT_IN*sin(angle_l-M_PI), dest_x_idx_right, dest_y_idx_right, mapWidth, resolution);

                double dx_left = abs(dest_x_idx_left - x_idx);
                double dy_left = abs(dest_y_idx_left - y_idx);
                double dx_right = abs(dest_x_idx_right - x_idx);
                double dy_right = abs(dest_y_idx_right - y_idx);

                double length_left;
                double length_right;

                if(dx_left >= dy_left){
                    length_left = dx_left;
                }
                else{
                    length_left = dy_left;
                }
                if(dx_right >= dy_right){
                    length_right = dx_right;
                }
                else{
                    length_right = dy_right;
                }

                dx_left = (dest_x_idx_left - x_idx)/length_left;
                dy_left = (dest_y_idx_left - y_idx)/length_left;
                dx_right = (dest_x_idx_right - x_idx)/length_right;
                dy_right = (dest_y_idx_right - y_idx)/length_right;

                double curr_x_left = x_idx + 0.49;
                double curr_y_left = y_idx + 0.49;
                double curr_x_right = x_idx + 0.49;
                double curr_y_right = y_idx + 0.49;

                int i_left = 1;
                int i_right = 1;

                double dist_left;
                double dist_right;

                double final_x_left;
                double final_y_left;
                double final_x_right;
                double final_y_right;

                while(i_left <= length_left){
                    if(gridMap((int)round(curr_y_left), (int)round(curr_x_left)) == 0 || i_left == length_left){
                        getCoordinate(final_x_left, final_y_left, (int)round(curr_x_left), (int)round(curr_y_left), mapWidth, resolution);
                        break;
                    }
                    curr_x_left += dx_left;
                    curr_y_left += dy_left;
                    i_left++;
                }
                while(i_right <= length_right){
                    if(gridMap(round(curr_y_right), round(curr_x_right)) == 0 || i_right == length_right){
                        getCoordinate(final_x_right, final_y_right, (int)round(curr_x_right), (int)round(curr_y_right), mapWidth, resolution);
                        break;
                    }
                    curr_x_right += dx_right;
                    curr_y_right += dy_right;
                    i_right++;
                }

                dist_left = sqrt(pow(x_cord - final_x_left,2) + pow(y_cord - final_y_left,2));
                dist_right = sqrt(pow(x_cord - final_x_right,2) + pow(y_cord - final_y_right,2));

                dist_grad_x[k] = coeff * -((EPSILON - dist_right)*(x_cord - final_x_left)/dist_left + (EPSILON - dist_left)*(x_cord - final_x_right)/dist_right);
                dist_grad_y[k] = coeff * -((EPSILON - dist_right)*(y_cord - final_y_left)/dist_left + (EPSILON - dist_left)*(y_cord - final_y_right)/dist_right);
                dist_hess_xx[k] = coeff * ((EPSILON - dist_right)*pow(x_cord - final_x_left, 2)/pow(dist_left, 3) + (EPSILON - dist_left)*pow(x_cord - final_x_right, 2)/pow(dist_right,3) - (EPSILON - dist_right)/dist_left - (EPSILON - dist_left)/dist_right + 2.0*(x_cord - final_x_right)*(x_cord - final_x_left)/(dist_right*dist_left));
                dist_hess_yy[k] = coeff * ((EPSILON - dist_right)*pow(y_cord - final_y_left, 2)/pow(dist_left, 3) + (EPSILON - dist_left)*pow(y_cord - final_y_right, 2)/pow(dist_right,3) - (EPSILON - dist_right)/dist_left - (EPSILON - dist_left)/dist_right + 2.0*(y_cord - final_y_right)*(y_cord - final_y_left)/(dist_right*dist_left));
                dist_hess_xy[k] = coeff * ((x_cord - final_x_left)*(y_cord - final_y_right)/(dist_left*dist_right) + (x_cord - final_x_right)*(y_cord - final_y_left)/(dist_left*dist_right) + (EPSILON - dist_right)*(x_cord - final_x_left)*(y_cord - final_y_left)/pow(dist_left, 3) + (EPSILON - dist_left)*(x_cord - final_x_right)*(y_cord - final_y_right)/pow(dist_right,3));

                dist[k] = coeff * (dist_left + EPSILON)*(dist_right + EPSILON);*/

                // Bresenham's
                double angle_l = th_cord + M_PI_2;

                int dest_x_idx_left;
                int dest_y_idx_left;
                getGridIndex(x_cord + TRACING_LIMIT_IN*cos(angle_l), y_cord + TRACING_LIMIT_IN*sin(angle_l), dest_x_idx_left, dest_y_idx_left, mapWidth, resolution);
                int dest_x_idx_right;
                int dest_y_idx_right;
                getGridIndex(x_cord + TRACING_LIMIT_IN*cos(angle_l-M_PI), y_cord + TRACING_LIMIT_IN*sin(angle_l-M_PI), dest_x_idx_right, dest_y_idx_right, mapWidth, resolution);

                double dx_left = abs(dest_x_idx_left - x_idx);
                double dy_left = abs(dest_y_idx_left - y_idx);
                double dx_right = abs(dest_x_idx_right - x_idx);
                double dy_right = abs(dest_y_idx_right - y_idx);

                double curr_x_left = x_idx;
                double curr_y_left = y_idx;
                double curr_x_right = x_idx;
                double curr_y_right = y_idx;

                int sx_left = x_idx > dest_x_idx_left ? -1 : 1;
                int sy_left = y_idx > dest_y_idx_left ? -1 : 1;
                int sx_right = x_idx > dest_x_idx_right ? -1 : 1;
                int sy_right = y_idx > dest_y_idx_right ? -1 : 1;

                bool no_left = false;
                bool no_right = false;

                if(dx_left > dy_left){
                    double err = dx_left / 2.0;
                    while(curr_x_left != dest_x_idx_left){
                        if(curr_y_left < 0 || curr_y_left > gridMap.rows()-1 || curr_x_left < 0 || curr_x_left > gridMap.cols()-1){
                            no_left = true;
                            break;
                        }
                        if(gridMap(round(curr_y_left), round(curr_x_left)) == 0){
                            // collision
                            getCoordinate(final_x_left[k], final_y_left[k], (int)round(curr_x_left), (int)round(curr_y_left), mapWidth, resolution);
                            break;
                        }
                        err -= dy_left;
                        if(err < 0){
                            curr_y_left += sy_left;
                            err += dx_left;
                        }
                        curr_x_left += sx_left;
                    }
                }
                else{
                    double err = dy_left / 2.0;
                    while(curr_y_left != dest_y_idx_left){
                        if(curr_y_left < 0 || curr_y_left > gridMap.rows()-1 || curr_x_left < 0 || curr_x_left > gridMap.cols()-1){
                            no_left = true;
                            break;
                        }
                        if(gridMap(round(curr_y_left), round(curr_x_left)) == 0){
                            // collision
                            getCoordinate(final_x_left[k], final_y_left[k], (int)round(curr_x_left), (int)round(curr_y_left), mapWidth, resolution);
                            break;
                        }
                        err -= dx_left;
                        if(err < 0){
                            curr_x_left += sx_left;
                            err += dy_left;
                        }
                        curr_y_left += sy_left;
                    }
                }
                if(no_left){
                    getCoordinate(final_x_left[k], final_y_left[k], (int)round(dest_x_idx_left), (int)round(dest_y_idx_left), mapWidth, resolution);
                }

                if(dx_right > dy_right){
                    double err = dx_right / 2.0;
                    while(curr_x_right != dest_x_idx_right){
                        if(curr_y_right < 0 || curr_y_right > gridMap.rows()-1 || curr_x_right < 0 || curr_x_right > gridMap.cols()-1){
                            no_right = true;
                            break;
                        }
                        if(gridMap(round(curr_y_right), round(curr_x_right)) == 0){
                            //collision
                            getCoordinate(final_x_right[k], final_y_right[k], (int)round(curr_x_right), (int)round(curr_y_right), mapWidth, resolution);
                            break;
                        }
                        err -= dy_right;
                        if(err < 0){
                            curr_y_right += sy_right;
                            err += dx_right;
                        }
                        curr_x_right += sx_right;
                    }
                }
                else{
                    double err = dy_right / 2.0;
                    while(curr_y_right != dest_y_idx_right){
                        if(curr_y_right < 0 || curr_y_right > gridMap.rows()-1 || curr_x_right < 0 || curr_x_right > gridMap.cols()-1){
                            no_right = true;
                            break;
                        }
                        if(gridMap(round(curr_y_right), round(curr_x_right)) == 0){
                            // collision
                            getCoordinate(final_x_right[k], final_y_right[k], (int)round(curr_x_right), (int)round(curr_y_right), mapWidth, resolution);
                            break;
                        }
                        err -= dx_right;
                        if(err < 0){
                            curr_x_right += sx_right;
                            err += dy_right;
                        }
                        curr_y_right += sy_right;
                    }
                }
                if(no_right){
                    getCoordinate(final_x_right[k], final_y_right[k], (int)round(dest_x_idx_right), (int)round(dest_y_idx_right), mapWidth, resolution);
                }

                double dist_left = sqrt(pow(x_cord - final_x_left[k],2) + pow(y_cord - final_y_left[k],2));
                double dist_right = sqrt(pow(x_cord - final_x_right[k],2) + pow(y_cord - final_y_right[k],2));

                dist[k] = (dist_left + EPSILON) * (dist_right + EPSILON);
                dist_grad_x[k] = (x_cord - final_x_left[k]) * dist_right / dist_left + (x_cord - final_x_right[k]) * dist_left / dist_right;
                dist_grad_y[k] = (y_cord - final_y_left[k]) * dist_right / dist_left + (y_cord - final_y_right[k]) * dist_left / dist_right;
                dist_hess_xx[k] = dist_right / dist_left + dist_left / dist_right - pow(final_x_left[k] - x_cord, 2) * dist_right / pow(dist_left, 3) - pow(final_x_right[k] - x_cord, 2) * dist_left / pow(dist_right, 3) + 2*(final_x_left[k] - x_cord) * (final_x_right[k] - x_cord) / (dist_left * dist_right);
                dist_hess_yy[k] = dist_right / dist_left + dist_left / dist_right - pow(final_y_left[k] - y_cord, 2) * dist_right / pow(dist_left, 3) - pow(final_y_right[k] - y_cord, 2) * dist_left / pow(dist_right, 3) + 2*(final_y_left[k] - y_cord) * (final_y_right[k] - y_cord) / (dist_left * dist_right);
                dist_hess_xy[k] = (final_x_left[k] - x_cord) * (final_y_right[k] - y_cord) / (dist_left * dist_right) + (final_x_right[k] - x_cord) * (final_y_left[k] - y_cord) / (dist_left * dist_right) - (final_x_left[k] - x_cord) * (final_y_left[k] - y_cord) * dist_right / pow(dist_left, 3) - (final_x_right[k] - x_cord) * (final_y_right[k] - y_cord) * dist_left / pow(dist_right, 3);
            }
            else{
                // outside
                // DDA
                /*double angle_l = th_cord + M_PI_2;

                int dest_x_idx_left;
                int dest_y_idx_left;
                getGridIndex(x_cord + TRACING_LIMIT_OUT*cos(angle_l), y_cord + TRACING_LIMIT_OUT*sin(angle_l), dest_x_idx_left, dest_y_idx_left, mapWidth, resolution);
                int dest_x_idx_right;
                int dest_y_idx_right;
                getGridIndex(x_cord + TRACING_LIMIT_OUT*cos(angle_l-M_PI), y_cord + TRACING_LIMIT_OUT*sin(angle_l-M_PI), dest_x_idx_right, dest_y_idx_right, mapWidth, resolution);

                double dx_left = abs(dest_x_idx_left - x_idx);
                double dy_left = abs(dest_y_idx_left - y_idx);
                double dx_right = abs(dest_x_idx_right - x_idx);
                double dy_right = abs(dest_y_idx_left - y_idx);

                double length_left;
                double length_right;

                if(dx_left >= dy_left){
                    length_left = dx_left;
                }
                else{
                    length_left = dy_left;
                }
                if(dx_right >= dy_right){
                    length_right = dx_right;
                }
                else{
                    length_right = dy_right;
                }

                dx_left = (dest_x_idx_left - x_idx)/length_left;
                dy_left = (dest_y_idx_left - y_idx)/length_left;
                dx_right = (dest_x_idx_right - x_idx)/length_right;
                dy_right = (dest_y_idx_right - y_idx)/length_right;

                double curr_x_left = x_idx + 0.49;
                double curr_y_left = y_idx + 0.49;
                double curr_x_right = x_idx + 0.49;
                double curr_y_right = y_idx + 0.49;

                int i_left = 1;
                int i_right = 1;

                double dist_left;
                double dist_right;

                double final_x_left;
                double final_y_left;
                double final_x_right;
                double final_y_right;

                bool no_left = false;
                bool no_right = false;

                while(i_left <= length_left){
                    if(gridMap(round(curr_y_left), round(curr_x_left)) != 0 || i_left == length_left){
                        getCoordinate(final_x_left, final_y_left, (int)round(curr_x_left), (int)round(curr_y_left), mapWidth, resolution);
                        if(i_left == length_left){
                            no_left = true;
                        }
                        break;
                    }
                    curr_x_left += dx_left;
                    curr_y_left += dy_left;
                    i_left++;
                }
                while(i_right <= length_right){
                    if(gridMap(round(curr_y_right), round(curr_x_right)) != 0 || i_right == length_right){
                        getCoordinate(final_x_right, final_y_right, (int)round(curr_x_right), (int)round(curr_y_right), mapWidth, resolution);
                        if(i_right == length_right){
                            no_right = true;
                        }
                        break;
                    }
                    curr_x_right += dx_right;
                    curr_y_right += dy_right;
                    i_right++;
                }

                dist_left = sqrt(pow(x_cord - final_x_left,2) + pow(y_cord - final_y_left,2));
                dist_right = sqrt(pow(x_cord - final_x_right,2) + pow(y_cord - final_y_right,2));

                dist[k] = coeff * -(dist_left - EPSILON)*(dist_right - EPSILON);

                if(no_left && no_right){
                    obs_grad_x[k] = 0.0;
                    obs_grad_y[k] = 0.0;
                    obs_hess_xx[k] = 0.0;
                    obs_hess_yy[k] = 0.0;
                    obs_hess_xy[k] = 0.0;
                }
                else if(no_left && !no_right){
                    dist_grad_x[k] = coeff * -dist_left*(x_cord - final_x_right)/dist_right;
                    dist_grad_y[k] = coeff * -dist_left*(y_cord - final_y_right)/dist_right;
                    dist_hess_xx[k] = coeff * (dist_left*pow(x_cord - final_x_right, 2)/pow(dist_right, 3) - dist_left/dist_right);
                    dist_hess_yy[k] = coeff * (dist_left*pow(y_cord - final_y_right, 2)/pow(dist_right, 3) - dist_left/dist_right);
                    dist_hess_xy[k] = coeff * (dist_left*(x_cord - final_x_right)*(y_cord - final_y_right)/pow(dist_right, 3));
                }
                else if(!no_left && no_right){
                    dist_grad_x[k] = coeff * -dist_right*(x_cord - final_x_left)/dist_left;
                    dist_grad_y[k] = coeff * -dist_right*(y_cord - final_y_left)/dist_left;
                    dist_hess_xx[k] = coeff * (dist_right*pow(x_cord - final_x_left, 2)/pow(dist_left, 3) - dist_right/dist_left);
                    dist_hess_yy[k] = coeff * (dist_right*pow(y_cord - final_y_left, 2)/pow(dist_left, 3) - dist_right/dist_left);
                    dist_hess_xy[k] = coeff * (dist_right*(x_cord - final_x_left)*(y_cord - final_y_left)/pow(dist_left, 3));
                }
                else{
                    dist_grad_x[k] = coeff * ((- dist_right)*(x_cord - final_x_left)/dist_left + (- dist_left)*(x_cord - final_x_right)/dist_right);
                    dist_grad_y[k] = coeff * ((- dist_right)*(y_cord - final_y_left)/dist_left + (- dist_left)*(y_cord - final_y_right)/dist_right);
                    dist_hess_xx[k] = coeff * (-(- dist_right)*pow(x_cord - final_x_left, 2)/pow(dist_left, 3) - (- dist_left)*pow(x_cord - final_x_right, 2)/pow(dist_right,3) + (- dist_right)/dist_left + (- dist_left)/dist_right - 2.0*(x_cord - final_x_right)*(x_cord - final_x_left)/(dist_right*dist_left));
                    dist_hess_yy[k] = coeff * (-(- dist_right)*pow(y_cord - final_y_left, 2)/pow(dist_left, 3) - (- dist_left)*pow(y_cord - final_y_right, 2)/pow(dist_right,3) + (- dist_right)/dist_left + (- dist_left)/dist_right - 2.0*(y_cord - final_y_right)*(y_cord - final_y_left)/(dist_right*dist_left));
                    dist_hess_xy[k] = coeff * (-(x_cord - final_x_left)*(y_cord - final_y_right)/(dist_left*dist_right) - (x_cord - final_x_right)*(y_cord - final_y_left)/(dist_left*dist_right) - (- dist_right)*(x_cord - final_x_left)*(y_cord - final_y_left)/pow(dist_left, 3) - (- dist_left)*(x_cord - final_x_right)*(y_cord - final_y_right)/pow(dist_right,3));
                }*/

                // Bresenham's
                double angle_l = th_cord + M_PI_2;

                int dest_x_idx_left;
                int dest_y_idx_left;
                getGridIndex(x_cord + TRACING_LIMIT_OUT*cos(angle_l), y_cord + TRACING_LIMIT_OUT*sin(angle_l), dest_x_idx_left, dest_y_idx_left, mapWidth, resolution);
                int dest_x_idx_right;
                int dest_y_idx_right;
                getGridIndex(x_cord + TRACING_LIMIT_OUT*cos(angle_l-M_PI), y_cord + TRACING_LIMIT_OUT*sin(angle_l-M_PI), dest_x_idx_right, dest_y_idx_right, mapWidth, resolution);

                double dx_left = abs(dest_x_idx_left - x_idx);
                double dy_left = abs(dest_y_idx_left - y_idx);
                double dx_right = abs(dest_x_idx_right - x_idx);
                double dy_right = abs(dest_y_idx_right - y_idx);

                double curr_x_left = x_idx;
                double curr_y_left = y_idx;
                double curr_x_right = x_idx;
                double curr_y_right = y_idx;

                double sx_left = x_idx > dest_x_idx_left ? -1 : 1;
                double sy_left = y_idx > dest_y_idx_left ? -1 : 1;
                double sx_right = x_idx > dest_x_idx_right ? -1 : 1;
                double sy_right = y_idx > dest_y_idx_right ? -1 : 1;

                bool no_left = false;
                bool no_right = false;

                if(dx_left > dy_left){
                    double err = dx_left / 2.0;
                    while(curr_x_left != dest_x_idx_left){
                        if(curr_y_left < 0 || curr_y_left > gridMap.rows()-1 || curr_x_left < 0 || curr_x_left > gridMap.cols()-1){
                            no_left = true;
                            break;
                        }
                        if(gridMap(round(curr_y_left), round(curr_x_left)) != 0){
                            // collision
                            getCoordinate(final_x_left[k], final_y_left[k], (int)round(curr_x_left), (int)round(curr_y_left), mapWidth, resolution);
                            break;
                        }
                        err -= dy_left;
                        if(err < 0){
                            curr_y_left += sy_left;
                            err += dx_left;
                        }
                        curr_x_left += sx_left;
                    }
                }
                else{
                    double err = dy_left / 2.0;
                    while(curr_y_left != dest_y_idx_left){
                        if(curr_y_left < 0 || curr_y_left > gridMap.rows()-1 || curr_x_left < 0 || curr_x_left > gridMap.cols()-1){
                            no_left = true;
                            break;
                        }
                        if(gridMap(round(curr_y_left), round(curr_x_left)) != 0){
                            // collision
                            getCoordinate(final_x_left[k], final_y_left[k], (int)round(curr_x_left), (int)round(curr_y_left), mapWidth, resolution);
                            break;
                        }
                        err -= dx_left;
                        if(err < 0){
                            curr_x_left += sx_left;
                            err += dy_left;
                        }
                        curr_y_left += sy_left;
                    }
                }
                if(no_left){
                    getCoordinate(final_x_left[k], final_y_left[k], (int)round(dest_x_idx_left), (int)round(dest_y_idx_left), mapWidth, resolution);
                }

                if(dx_right > dy_right){
                    double err = dx_right / 2.0;
                    while(curr_x_right != dest_x_idx_right){
                        if(curr_y_right < 0 || curr_y_right > gridMap.rows()-1 || curr_x_right < 0 || curr_x_right > gridMap.cols()-1){
                            no_right = true;
                            break;
                        }
                        if(gridMap(round(curr_y_right), round(curr_x_right)) != 0){
                            // collision
                            getCoordinate(final_x_right[k], final_y_right[k], (int)round(curr_x_right), (int)round(curr_y_right), mapWidth, resolution);
                            break;
                        }
                        err -= dy_right;
                        if(err < 0){
                            curr_y_right += sy_right;
                            err += dx_right;
                        }
                        curr_x_right += sx_right;
                    }
                }
                else{
                    double err = dy_right / 2.0;
                    while(curr_y_right != dest_y_idx_right){
                        if(curr_y_right < 0 || curr_y_right > gridMap.rows()-1 || curr_x_right < 0 || curr_x_right > gridMap.cols()-1){
                            no_right = true;
                            break;
                        }
                        if(gridMap(round(curr_y_right), round(curr_x_right)) != 0){
                            // collision
                            getCoordinate(final_x_right[k], final_y_right[k], (int)round(curr_x_right), (int)round(curr_y_right), mapWidth, resolution);
                            break;
                        }
                        err -= dx_right;
                        if(err < 0){
                            curr_x_right += sx_right;
                            err += dy_right;
                        }
                        curr_y_right += sy_right;
                    }
                }
                if(no_right){
                    getCoordinate(final_x_right[k], final_y_right[k], (int)round(dest_x_idx_right), (int)round(dest_y_idx_right), mapWidth, resolution);
                }

                double dist_left = sqrt(pow(final_x_left[k] - x_cord,2) + pow(final_y_left[k] - y_cord,2));
                double dist_right = sqrt(pow(final_x_right[k] - x_cord,2) + pow(final_y_right[k] - y_cord,2));

                dist[k] = -(dist_left - EPSILON) * (dist_right - EPSILON);

                if(no_left && no_right){
                    obs_grad_x[k] = 0.0;
                    obs_grad_y[k] = 0.0;
                    obs_hess_xx[k] = 0.0;
                    obs_hess_yy[k] = 0.0;
                    obs_hess_xy[k] = 0.0;
                }
                else if(no_left && !no_right){
                    dist_grad_x[k] = -dist_left*(x_cord - final_x_right[k])/dist_right;
                    dist_grad_y[k] = -dist_left*(y_cord - final_y_right[k])/dist_right;
                    dist_hess_xx[k] = (dist_left*pow(x_cord - final_x_right[k], 2)/pow(dist_right, 3) - dist_left/dist_right);
                    dist_hess_yy[k] = (dist_left*pow(y_cord - final_y_right[k], 2)/pow(dist_right, 3) - dist_left/dist_right);
                    dist_hess_xy[k] = (dist_left*(x_cord - final_x_right[k])*(y_cord - final_y_right[k])/pow(dist_right, 3));
                }
                else if(!no_left && no_right){
                    dist_grad_x[k] = -dist_right*(x_cord - final_x_left[k])/dist_left;
                    dist_grad_y[k] = -dist_right*(y_cord - final_y_left[k])/dist_left;
                    dist_hess_xx[k] = (dist_right*pow(x_cord - final_x_left[k], 2)/pow(dist_left, 3) - dist_right/dist_left);
                    dist_hess_yy[k] = (dist_right*pow(y_cord - final_y_left[k], 2)/pow(dist_left, 3) - dist_right/dist_left);
                    dist_hess_xy[k] = (dist_right*(x_cord - final_x_left[k])*(y_cord - final_y_left[k])/pow(dist_left, 3));
                }
                else{
                    dist_grad_x[k] = ((- dist_right)*(x_cord - final_x_left[k])/dist_left + (- dist_left)*(x_cord - final_x_right[k])/dist_right);
                    dist_grad_y[k] = ((- dist_right)*(y_cord - final_y_left[k])/dist_left + (- dist_left)*(y_cord - final_y_right[k])/dist_right);
                    dist_hess_xx[k] = (-(- dist_right)*pow(x_cord - final_x_left[k], 2)/pow(dist_left, 3) - (- dist_left)*pow(x_cord - final_x_right[k], 2)/pow(dist_right,3) + (- dist_right)/dist_left + (- dist_left)/dist_right - 2.0*(x_cord - final_x_right[k])*(x_cord - final_x_left[k])/(dist_right*dist_left));
                    dist_hess_yy[k] = (-(- dist_right)*pow(y_cord - final_y_left[k], 2)/pow(dist_left, 3) - (- dist_left)*pow(y_cord - final_y_right[k], 2)/pow(dist_right,3) + (- dist_right)/dist_left + (- dist_left)/dist_right - 2.0*(y_cord - final_y_right[k])*(y_cord - final_y_left[k])/(dist_right*dist_left));
                    dist_hess_xy[k] = (-(x_cord - final_x_left[k])*(y_cord - final_y_right[k])/(dist_left*dist_right) - (x_cord - final_x_right[k])*(y_cord - final_y_left[k])/(dist_left*dist_right) - (- dist_right)*(x_cord - final_x_left[k])*(y_cord - final_y_left[k])/pow(dist_left, 3) - (- dist_left)*(x_cord - final_x_right[k])*(y_cord - final_y_right[k])/pow(dist_right,3));
                }
            }
        }
    }

    double c(double dist){
        if(dist >= 0){
            return pow(dist, 2);
        }
        else{
            return 0.0;
        }
    }

    double dc(double dist, double ddist){
        if(dist >= 0){
            return 2.0*dist*ddist;
        }
        else{
            return 0.0;
        }
    }

    double ddc(double dist, double ddist_1, double ddist_2, double dddist){
        if(dist >= 0.0){
            return 2.0 * (ddist_1*ddist_2 + dist*dddist);
        }
        else{
            return 0.0;
        }
    }

    void getGridIndex(const double& x, const double& y, int& idx_x, int& idx_y, double map_size, double resolution){
        idx_x = (int)((x + map_size/2.0)/resolution);
        idx_y = (int)((y + map_size/2.0)/resolution);
    }

    void getCoordinate(double& x, double& y, const int idx_x, int idx_y, double map_size, double resolution){
        x = (idx_x * resolution + resolution/2.0) - map_size/2.0;
        y = (idx_y * resolution + resolution/2.0) - map_size/2.0;
    }

    bool getGradient(int n, const double* x, bool new_x, double* grad_f){
        int c = 0;

        double h = x[0]/N;

        double tfSum = tf_coeff;
        if(comfort_coeff > 0){
            for(int k = 0; k < N; k++){
                tfSum += comfort_grad_tf[k];
            }
        }

        grad_f[c++] = tfSum;

        // x
        for(int k = 0; k < N+1; k++){
            grad_f[c++] = obs_grad_x[k];
        }

        // y
        for(int k = 0; k < N+1; k++){
            grad_f[c++] = obs_grad_y[k];
        }

        // v
        grad_f[c++] = comfort_grad_v[0];
        for(int k = 1; k < N; k++){
            grad_f[c++] = comfort_grad_v[k] + comfort_grad_v_n[k-1];
        }
        grad_f[c++] = comfort_grad_v_n[N-1];

        // th
        for(int k = 0; k < N+1; k++){
            grad_f[c++] = 0.0;
        }

        // phi
        grad_f[c++] = comfort_grad_phi[0];
        for(int k = 1; k < N; k++){
            grad_f[c++] = comfort_grad_phi[k] + comfort_grad_phi_n[k-1];
        }
        grad_f[c++] = comfort_grad_phi_n[N-1];

        // a
        grad_f[c++] = comfort_grad_a[0];
        for(int k = 1; k < N; k++){
            grad_f[c++] = comfort_grad_a[k] + comfort_grad_a_n[k-1];
        }
        grad_f[c++] = comfort_grad_a_n[N-1];

        for(int k = 0; k < N+1; k++){
            grad_f[c++] = 0.0;
        }

        assert(c == n);
        return c == n;
    }

    bool getConstraints(int n, const double* x, bool new_x, int m, double* g){
        double h = x[0]/N;

        int c = 0;
        // dynamic constraints for x
        for(int k = 0; k < N; k++){
            g[c++] = x[1 + 0*(N+1) + (k+1)] - x[1 + 0*(N+1) + k] - (h/2.0)*(x[1 + 2*(N+1) + (k+1)] * cos(x[1 + 3*(N+1) + (k+1)]) + x[1 + 2*(N+1) + k] * cos(x[1 + 3*(N+1) + k]));
        }
        // dynamic constraints for y
        for(int k = 0; k < N; k++){
            g[c++] = x[1 + 1*(N+1) + (k+1)] - x[1 + 1*(N+1) + k] - (h/2.0)*(x[1 + 2*(N+1) + (k+1)] * sin(x[1 + 3*(N+1) + (k+1)]) + x[1 + 2*(N+1) + k] * sin(x[1 + 3*(N+1) + k]));
        }
        // dynamic constraints for v
        for(int k = 0; k < N; k++){
            g[c++] = x[1 + 2*(N+1) + (k+1)] - x[1 + 2*(N+1) + k] - (h/2.0)*(x[1 + 5*(N+1) + (k+1)] + x[1 + 5*(N+1) + k]);
        }
        // dynamic constraints for th
        for(int k = 0; k < N; k++){
            g[c++] = x[1 + 3*(N+1) + (k+1)] - x[1 + 3*(N+1) + k] - (h/2.0)*(x[1 + 2*(N+1) + (k+1)]*tan(x[1 + 4*(N+1) + (k+1)])/l + x[1 + 2*(N+1) + k]*tan(x[1 + 4*(N+1) + k])/l);
        }
        // dynamic constraints for phi
        for(int k = 0; k < N; k++){
            g[c++] = x[1 + 4*(N+1) + (k+1)] - x[1 + 4*(N+1) + k] - (h/2.0)*(x[1 + 6*(N+1) + (k+1)] + x[1 + 6*(N+1) + k]);
        }

        // boundary constraints for start
        for(int q = 0; q < stateDim; q++){
            g[c++] = x[1 + q*(N+1) + 0] - startState(q, 0);
        }
        // boundary constraints for goal
        for(int q = 0; q < stateDim; q++){
            g[c++] = x[1 + q*(N+1) + N] - goalState(q, 0);
        }

        // comfort constraints
        for(int i = 0; i < N+1; i++){
            g[c++] = pow(x[1 + 5*(N+1) + i],2) + pow(x[1 + 2*(N+1) + i],4)*pow(tan(x[1 + 4*(N+1) + i]),2)/pow(l,2);
        }

        return true;
    }

    bool getJacobian(int n, const double* x, bool new_x, int m, int nele_jac, int* iRow, int* jCol, double* values){
        if (values == NULL) {

            int c = 0;
            // dynamic constraints of x
            for (int i = 0; i < N; i++) {
                iRow[c] = i;
                jCol[c++] = 0;

                iRow[c] = i;
                jCol[c++] = 1 + 0 * (N + 1) + i;

                iRow[c] = i;
                jCol[c++] = 1 + 0 * (N + 1) + i + 1;

                iRow[c] = i;
                jCol[c++] = 1 + 2 * (N + 1) + i;

                iRow[c] = i;
                jCol[c++] = 1 + 2 * (N + 1) + i + 1;

                iRow[c] = i;
                jCol[c++] = 1 + 3 * (N + 1) + i;

                iRow[c] = i;
                jCol[c++] = 1 + 3 * (N + 1) + i + 1;
            }

            // dynamic constraints of y
            for (int i = 0; i < N; i++) {
                iRow[c] = N + i;
                jCol[c++] = 0;

                iRow[c] = N + i;
                jCol[c++] = 1 + 1 * (N + 1) + i;

                iRow[c] = N + i;
                jCol[c++] = 1 + 1 * (N + 1) + i + 1;

                iRow[c] = N + i;
                jCol[c++] = 1 + 2 * (N + 1) + i;

                iRow[c] = N + i;
                jCol[c++] = 1 + 2 * (N + 1) + i + 1;

                iRow[c] = N + i;
                jCol[c++] = 1 + 3 * (N + 1) + i;

                iRow[c] = N + i;
                jCol[c++] = 1 + 3 * (N + 1) + i + 1;
            }

            // dynamic constraints of v
            for (int i = 0; i < N; i++) {
                iRow[c] = 2 * N + i;
                jCol[c++] = 0;

                iRow[c] = 2 * N + i;
                jCol[c++] = 1 + 2 * (N + 1) + i;

                iRow[c] = 2 * N + i;
                jCol[c++] = 1 + 2 * (N + 1) + i + 1;

                iRow[c] = 2 * N + i;
                jCol[c++] = 1 + 5 * (N + 1) + i;

                iRow[c] = 2 * N + i;
                jCol[c++] = 1 + 5 * (N + 1) + i + 1;
            }

            // dynamic constraints of th
            for (int i = 0; i < N; i++) {
                iRow[c] = 3 * N + i;
                jCol[c++] = 0;

                iRow[c] = 3 * N + i;
                jCol[c++] = 1 + 2 * (N + 1) + i;

                iRow[c] = 3 * N + i;
                jCol[c++] = 1 + 2 * (N + 1) + i + 1;

                iRow[c] = 3 * N + i;
                jCol[c++] = 1 + 3 * (N + 1) + i;

                iRow[c] = 3 * N + i;
                jCol[c++] = 1 + 3 * (N + 1) + i + 1;

                iRow[c] = 3 * N + i;
                jCol[c++] = 1 + 4 * (N + 1) + i;

                iRow[c] = 3 * N + i;
                jCol[c++] = 1 + 4 * (N + 1) + i + 1;
            }

            // dynamic constraints of del
            for (int i = 0; i < N; i++) {
                iRow[c] = 4 * N + i;
                jCol[c++] = 0;

                iRow[c] = 4 * N + i;
                jCol[c++] = 1 + 4 * (N + 1) + i;

                iRow[c] = 4 * N + i;
                jCol[c++] = 1 + 4 * (N + 1) + i + 1;

                iRow[c] = 4 * N + i;
                jCol[c++] = 1 + 6 * (N + 1) + i;

                iRow[c] = 4 * N + i;
                jCol[c++] = 1 + 6 * (N + 1) + i + 1;
            }

            // boundary condition of start
            iRow[c] = 5 * N + 0;
            jCol[c++] = 1 + 0 * (N + 1) + 0;

            iRow[c] = 5 * N + 1;
            jCol[c++] = 1 + 1 * (N + 1) + 0;

            iRow[c] = 5 * N + 2;
            jCol[c++] = 1 + 2 * (N + 1) + 0;

            iRow[c] = 5 * N + 3;
            jCol[c++] = 1 + 3 * (N + 1) + 0;

            iRow[c] = 5 * N + 4;
            jCol[c++] = 1 + 4 * (N + 1) + 0;

            iRow[c] = 5 * N + 5;
            jCol[c++] = 1 + 0 * (N + 1) + N;

            iRow[c] = 5 * N + 6;
            jCol[c++] = 1 + 1 * (N + 1) + N;

            iRow[c] = 5 * N + 7;
            jCol[c++] = 1 + 2 * (N + 1) + N;

            iRow[c] = 5 * N + 8;
            jCol[c++] = 1 + 3 * (N + 1) + N;

            iRow[c] = 5 * N + 9;
            jCol[c++] = 1 + 4 * (N + 1) + N;

            for(int i = 0; i < N+1; i++){
                iRow[c] = 5 * N + 10 + i;
                jCol[c++] = 1 + 2 * (N + 1) + i;

                iRow[c] = 5 * N + 10 + i;
                jCol[c++] = 1 + 4 * (N + 1) + i;

                iRow[c] = 5 * N + 10 + i;
                jCol[c++] = 1 + 5 * (N + 1) + i;
            }
        }
        else{
            int c = 0;
            Number h = x[0]/(double)N;
            // dynamic constraints of x
            for(int k = 0; k < N; k++){
                values[c++] = -(x[1 + 2*(N+1) + (k+1)]*cos(x[1 + 3*(N+1) + (k+1)]) + x[1 + 2*(N+1) + k]*cos(x[1 + 3*(N+1) + k]))/(2.0*N);

                values[c++] = -1;
                values[c++] = 1;

                values[c++] = -h/2.0*cos(x[1 + 3*(N+1) + k]);
                values[c++] = -h/2.0*cos(x[1 + 3*(N+1) + (k+1)]);

                values[c++] = h/2.0*x[1 + 2*(N+1) + k]*sin(x[1 + 3*(N+1) + k]);
                values[c++] = h/2.0*x[1 + 2*(N+1) + (k+1)]*sin(x[1 + 3*(N+1) + (k+1)]);
            }

            // dynamic constraints of y
            for(int  k = 0; k < N; k++){
                values[c++] = -(x[1 + 2*(N+1) + (k+1)]*sin(x[1 + 3*(N+1) + (k+1)]) + x[1 + 2*(N+1) + k]*sin(x[1 + 3*(N+1) + k]))/(2.0*N);

                values[c++] = -1;
                values[c++] = 1;

                values[c++] = -h/2.0*sin(x[1 + 3*(N+1) + k]);
                values[c++] = -h/2.0*sin(x[1 + 3*(N+1) + (k+1)]);

                values[c++] = -h/2.0*x[1 + 2*(N+1) + k]*cos(x[1 + 3*(N+1) + k]);
                values[c++] = -h/2.0*x[1 + 2*(N+1) + (k+1)]*cos(x[1 + 3*(N+1) + (k+1)]);
            }

            // dynamic constraints of v
            for(int k = 0; k < N; k++){
                values[c++] = -(x[1 + 5*(N+1) + (k+1)] + x[1 + 5*(N+1) + k])/(2.0*N);

                values[c++] = -1;
                values[c++] = 1;

                values[c++] = -h/2.0;
                values[c++] = -h/2.0;
            }

            // dynamic constraints of th
            for(int k = 0; k < N; k++){
                values[c++] = -(x[1 + 2*(N+1) + (k+1)]/l*tan(x[1 + 4*(N+1) + (k+1)]) + x[1 + 2*(N+1) + k]/l*tan(x[1 + 4*(N+1) + k]))/(2.0*N);

                values[c++] = (-h*tan(x[1 + 4*(N+1) + k]))/(2.0*l);
                values[c++] = (-h*tan(x[1 + 4*(N+1) + (k+1)]))/(2.0*l);

                values[c++] = -1;
                values[c++] = 1;

                values[c++] = (-h*x[1 + 2*(N+1) + k] * pow(1.0/cos(x[1 + 4*(N+1) + k]),2))/(2.0*l);
                values[c++] = (-h*x[1 + 2*(N+1) + (k+1)] * pow(1.0/cos(x[1 + 4*(N+1) + (k+1)]),2))/(2.0*l);
            }

            // dynamic constraints of del
            for(int k = 0; k < N; k++){
                values[c++] = -(x[1 + 6*(N+1) + (k+1)] + x[1 + 6*(N+1) + k])/(2.0*N);

                values[c++] = -1;
                values[c++] = 1;

                values[c++] = -h/2.0;
                values[c++] = -h/2.0;
            }

            values[c++] = 1;

            values[c++] = 1;

            values[c++] = 1;

            values[c++] = 1;

            values[c++] = 1;

            values[c++] = 1;

            values[c++] = 1;

            values[c++] = 1;

            values[c++] = 1;

            values[c++] = 1;

            for(int k = 0; k < N+1; k++){
                values[c++] = 4.0*pow(x[1 + 2*(N+1) + k],3)*pow(tan(x[1 + 4*(N+1)+ k]),2)/pow(l,2);

                values[c++] = 2.0*pow(x[1 + 2*(N+1) + k],4)*tan(x[1 + 4*(N+1) + k])*(pow(tan(x[1 + 4*(N+1) + k]),2) + 1)/pow(l,2);

                values[c++] = 2.0*x[1 + 5*(N+1) + k];
            }
        }

        return true;
    }

    bool getHessian(int n, const double* x, bool new_x, double obj_factor, int m, const double* lambda, bool new_lambda, int nele_hess, int* iRow, int* jCol, double* values){
        if (values == NULL){

            int c = 0;

            // dx
            for(int i = 0; i < N+1; i++){
                iRow[c] = 1 + 0*(N+1) + i;
                jCol[c++] = 1 + 0*(N+1) + i;
            }
            // dy
            for(int i = 0; i < N+1; i++){
                iRow[c] = 1 + 1*(N+1) + i;
                jCol[c++] = 1 + 0*(N+1) + i;

                iRow[c] = 1 + 1*(N+1) + i;
                jCol[c++] = 1 + 1*(N+1) + i;
            }

            // dv
            for(int i = 0; i < N+1; i++){
                iRow[c] = 1 + 2*(N+1) + i;
                jCol[c++] = 0;

                iRow[c] = 1 + 2*(N+1) + i;
                jCol[c++] = 1 + 2*(N+1) + i;
            }

            // dth
            for(int i = 0; i < N+1; i++){
                iRow[c] = 1 + 3*(N+1) + i;
                jCol[c++] = 0;

                iRow[c] = 1 + 3*(N+1) + i;
                jCol[c++] = 1 + 2*(N+1) + i;

                iRow[c] = 1 + 3*(N+1) + i;
                jCol[c++] = 1 + 3*(N+1) + i;
            }

            // dphi
            for(int i = 0; i < N+1; i++){
                iRow[c] = 1 + 4*(N+1) + i;
                jCol[c++] = 0;

                iRow[c] = 1 + 4*(N+1) + i;
                jCol[c++] = 1 + 2*(N+1) + i;

                iRow[c] = 1 + 4*(N+1) + i;
                jCol[c++] = 1 + 4*(N+1) + i;
            }

            // da
            for(int i = 0; i < N+1; i++){
                iRow[c] = 1 + 5*(N+1) + i;
                jCol[c++] = 0;

                iRow[c] = 1 + 5*(N+1) + i;
                jCol[c++] = 1 + 5*(N+1) + i;
            }

            // dw
            for(int i = 0; i < N+1; i++){
                iRow[c] = 1 + 6*(N+1) + i;
                jCol[c++] = 0;
            }

        }
        else{
            int c = 0;

            //dx
            for(int i = 0; i < N+1; i++){
                values[c++] = obj_factor*obs_hess_xx[i];
            }

            //dy
            for(int i = 0; i < N+1; i++){
                values[c++] = obj_factor*obs_hess_xy[i];

                values[c++] = obj_factor*obs_hess_yy[i];
            }

            // dv
            values[c++] = obj_factor*(comfort_hess_tfv[0]) - lambda[1*N]*sin(x[1 + 4*(N+1) + 0])/(2.0*N) -lambda[4*N]*tan(x[1 + 4*(N+1) + 0])/(2.0*N*l) - lambda[0*N]*cos(x[1 + 4*(N+1) + 0])/(2.0*N);
            values[c++] = obj_factor*(comfort_hess_vv[0]);
            for(int i = 1; i < N; i++){
                values[c++] = obj_factor*(comfort_hess_tfv[i] + comfort_hess_tfv_n[i-1]) - lambda[0*N+i]*cos(x[1 + 4*(N+1) + i])/(2.0*N) - lambda[1*N+(i-1)]*sin(x[1 + 4*(N+1) + i])/(2.0*N) -lambda[1*N+i]*sin(x[1 + 4*(N+1) + i])/(2.0*N) -lambda[4*N + (i-1)]*tan(x[1 + 4*(N+1) + i])/(2.0*N*l) - lambda[4*N+i]*tan(x[1 + 4*(N+1) + i])/(2.0*N*l) - lambda[0*N + (i-1)]*cos(x[1 + 4*(N+1) + i])/(2.0*N);
                values[c++] = obj_factor*(comfort_hess_vv[i] + comfort_hess_vv_n[i-1]);
            }
            values[c++] = obj_factor*(comfort_hess_tfv_n[N-1]) - lambda[1*N + N-1]*sin(x[1 + 4*(N+1) + N])/(2.0*N) -lambda[4*N + N-1]*tan(x[1 + 4*(N+1) + N])/(2.0*N*l) - lambda[0*N + N-1]*cos(x[1 + 4*(N+1) + N])/(2.0*N);
            values[c++] = obj_factor*(comfort_hess_vv_n[N-1]);

            //dth
            values[c++] = lambda[0*N]*x[1 + 2*(N+1) + 0]*sin(x[1 + 3*(N+1) + 0])/(2.0*N) - lambda[1*N]*x[1 + 2*(N+1) + 0]*cos(x[1 + 3*(N+1) + 0])/(2.0*N);
            values[c++] = lambda[0*N]*x[0]*sin(x[1 + 3*(N+1) + 0])/(2.0*N) - lambda[1*N]*x[0]*cos(x[1 + 3*(N+1) + 0])/(2.0*N);
            values[c++] = lambda[0*N]*x[0]*x[1 + 2*(N+1) + 0]*cos(x[1 + 3*(N+1) + 0])/(2.0*N) + lambda[1*N]*x[0]*x[1 + 2*(N+1) + 0]*sin(x[1 + 3*(N+1) + 0])/(2.0*N);
            for(int i = 1; i < N; i++){
                values[c++] = lambda[0*N + (i-1)]*x[1 + 2*(N+1) + i]*sin(x[1 + 3*(N+1) + i])/(2.0*N) - lambda[1*N + i]*x[1 + 2*(N+1) + i]*cos(x[1 + 3*(N+1) + i])/(2.0*N) - lambda[1*N + (i-1)]*x[1 + 2*(N+1) + i]*cos(x[1 + 3*(N+1) + i])/(2.0*N) + lambda[0*N + (i-1)]*x[1 + 2*(N+1) + i]*sin(x[1 + 3*(N+1) + i])/(2.0*N);
                values[c++] = lambda[0*N + (i-1)]*x[0]*sin(x[1 + 3*(N+1) + i])/(2.0*N) - lambda[1*N + i]*x[0]*cos(x[1 + 3*(N+1) + i])/(2.0*N) - lambda[1*N + (i-1)]*x[0]*cos(x[1 + 3*(N+1) + i])/(2.0*N) + lambda[0*N + i]*x[0]*sin(x[1 + 3*(N+1) + i])/(2.0*N);
                values[c++] = lambda[0*N + (i-1)]*x[0]*x[1 + 2*(N+1) + i]*cos(x[1 + 3*(N+1) + i])/(2.0*N) + lambda[0*N + i]*x[0]*x[1 + 2*(N+1) + i]*cos(x[1 + 3*(N+1) + i])/(2.0*N) + lambda[1*N + (i-1)]*x[0]*x[1 + 2*(N+1) + i]*sin(x[1 + 3*(N+1) + i])/(2.0*N) + lambda[1*N + i]*x[0]*x[1 + 2*(N+1) + i]*sin(x[1 + 3*(N+1) + i])/(2.0*N);
            }
            values[c++] = lambda[0*N + (N-1)]*x[1 + 2*(N+1) + N]*sin(x[1 + 3*(N+1) + N])/(2.0*N) - lambda[1*N + (N-1)]*x[1 + 2*(N+1) + N]*cos(x[1 + 3*(N+1) + N])/(2.0*N);
            values[c++] = lambda[0*N + (N-1)]*x[0]*sin(x[1 + 3*(N+1) + N])/(2.0*N) - lambda[1*N + (N-1)]*x[0]*cos(x[1 + 3*(N+1) + N])/(2.0*N);
            values[c++] = lambda[0*N + (N-1)]*x[0]*x[1 + 2*(N+1) + N]*cos(x[1 + 3*(N+1) + N])/(2.0*N) + lambda[1*N + (N-1)]*x[0]*x[1 + 2*(N+1) + N]*sin(x[1 + 3*(N+1) + N])/(2.0*N);

            // dphi
            values[c++] = obj_factor*(comfort_hess_tfphi[0]) - lambda[3*N]*x[1 + 2*(N+1) + 0]*(pow(tan(x[1 + 4*(N+1) + 0]),2) + 1)/(2.0*N*l);
            values[c++] = obj_factor*(comfort_hess_vphi[0]) - lambda[3*N]*x[0]*(pow(tan(x[1 + 4*(N+1) + 0]),2) + 1)/(2.0*N*l);
            values[c++] = obj_factor*(comfort_hess_phiphi[0]) - lambda[3*N]*x[0]*x[1 + 2*(N+1) + 0]*tan(x[1 + 4*(N+1) + 0])*(pow(tan(x[1 + 4*(N+1) + 0]),2) + 1)/(N*l);
            for(int i = 1; i < N; i++){
                values[c++] = obj_factor*(comfort_hess_tfphi[i] + comfort_hess_tfphi_n[i-1]) - lambda[3*N + i]*x[1 + 2*(N+1) + i]*(pow(tan(x[1 + 4*(N+1) + i]),2) + 1)/(2.0*N*l) - lambda[3*N + (i-1)]*x[1 + 2*(N+1) + i]*(pow(tan(x[1 + 4*(N+1) + i]),2) + 1)/(2.0*N*l);
                values[c++] = obj_factor*(comfort_hess_vphi[i] + comfort_hess_vphi_n[i-1]) - lambda[3*N + i]*x[0]*(pow(tan(x[1 + 4*(N+1) + i]),2) + 1)/(2.0*N*l) - lambda[3*N + (i-1)]*x[0]*(pow(tan(x[1 + 4*(N+1) + i]),2) + 1)/(2.0*N*l);
                values[c++] = obj_factor*(comfort_hess_phiphi[i] + comfort_hess_phiphi_n[i-1]) - lambda[3*N + (i-1)]*x[0]*x[1 + 2*(N+1) + i]*tan(x[1 + 4*(N+1) + i])*(pow(tan(x[1 + 4*(N+1) + i]),2) + 1)/(N*l) - lambda[3*N + i]*x[0]*x[1 + 2*(N+1) + i]*tan(x[1 + 4*(N+1) + i])*(pow(tan(x[1 + 4*(N+1) + i]),2) + 1)/(N*l);
            }
            values[c++] = obj_factor*(comfort_hess_tfphi_n[N-1]) - lambda[3*N + (N-1)]*x[1 + 2*(N+1) + N]*(pow(tan(x[1 + 4*(N+1) + N]),2) + 1)/(2.0*N*l);
            values[c++] = obj_factor*(comfort_hess_vphi_n[N-1]) - lambda[3*N + (N-1)]*x[0]*(pow(tan(x[1 + 4*(N+1) + N]),2) + 1)/(2.0*N*l);
            values[c++] = obj_factor*(comfort_hess_phiphi_n[N-1]) - lambda[3*N + (N-1)]*x[0]*x[1 + 2*(N+1) + N]*tan(x[1 + 4*(N+1) + N])*(pow(tan(x[1 + 4*(N+1) + N]),2) + 1)/(N*l);

            // da
            values[c++] = obj_factor*(comfort_hess_tfa[0]) - lambda[2*N]/(2.0*N);
            values[c++] = obj_factor*(comfort_hess_aa[0]);
            for(int i = 1; i < N; i++){
                values[c++] = obj_factor*(comfort_hess_tfa[i] + comfort_hess_tfa_n[i-1]) - lambda[2*N + i]/(2.0*N) - lambda[2*N + (i-1)]/(2.0*N);
                values[c++] = obj_factor*(comfort_hess_aa[i] + comfort_hess_aa_n[i-1]);
            }
            values[c++] = obj_factor*(comfort_hess_tfa_n[N-1]) - lambda[2*N + (N-1)]/(2.0*N);
            values[c++] = obj_factor*(comfort_hess_aa_n[N-1]);

            //dw
            values[c++] = -lambda[4*N]/(2.0*N);
            for(int i = 1; i < N; i++){
                values[c++] = -lambda[4*N + (i-1)]/(2.0*N) - lambda[4*N + i]/(2.0*N);
            }
            values[c++] = -lambda[4*N + (N-1)]/(2.0*N);

            std_msgs::Float64MultiArray msg;
            msg.layout.data_offset = N;
            for(int i = 0; i < getVariablesSize(); i++){
                msg.data.push_back(x[i]);
            }
            pubTrajectory.publish(msg);

            std_msgs::Float64MultiArray msg2;
            msg2.layout.data_offset = 4*(N+1);
            for(int i = 0; i < N+1; i++){
                msg2.data.push_back(final_x_left[i]);
            }
            for(int i = 0; i < N+1; i++){
                msg2.data.push_back(final_y_left[i]);
            }
            for(int i = 0; i < N+1; i++){
                msg2.data.push_back(final_x_right[i]);
            }
            for(int i = 0; i < N+1; i++){
                msg2.data.push_back(final_y_right[i]);
            }
            pubDetectedObstacle.publish(msg2);
        }

        return true;
    }

    void finalizeSolution(int n, const double* x){
        finalSolution = new double[n]();
        for(int i = 0; i < n; i++){
            finalSolution[i] = x[i];
        }
    }

    double* finalSolution;

private:
    int N;
    Eigen::Matrix<double, stateDim+actionDim, 1> startState;
    Eigen::Matrix<double, stateDim+actionDim, 1> goalState;
    double l = 0.324;
    ros::ServiceClient client;
    ros::Publisher pubTrajectory;
    ros::Publisher pubDetectedObstacle;
    double resolution;
    double mapWidth;
    Eigen::MatrixXi gridMap;
    Eigen::MatrixXd hermite_S;
    Eigen::MatrixXd hermite_h;
    double hermite_coeff = 1.0;
    double tf_coeff = 1.0;
    double obs_coeff = 100.0;
    double comfort_coeff = 1.0;
    double Cmax = 1.63;
    double TRACING_LIMIT_IN = 1.0;
    double TRACING_LIMIT_OUT = 1.0;
    double EPSILON = 1.0;
    double* final_x_left;
    double* final_y_left;
    double* final_x_right;
    double* final_y_right;
    double* obs_grad_x;
    double* obs_grad_y;
    double* obs_hess_xx;
    double* obs_hess_yy;
    double* obs_hess_xy;
    double* dist;
    double* dist_grad_x;
    double* dist_grad_y;
    double* dist_hess_xx;
    double* dist_hess_yy;
    double* dist_hess_xy;
    double* comfort_grad_tf;
    double* comfort_grad_v;
    double* comfort_grad_phi;
    double* comfort_grad_a;
    double* comfort_grad_v_n;
    double* comfort_grad_phi_n;
    double* comfort_grad_a_n;
    double* comfort_hess_tfv;
    double* comfort_hess_tfphi;
    double* comfort_hess_tfa;
    double* comfort_hess_vv;
    double* comfort_hess_vphi;
    double* comfort_hess_phiphi;
    double* comfort_hess_aa;
    double* comfort_hess_tfv_n;
    double* comfort_hess_tfphi_n;
    double* comfort_hess_tfa_n;
    double* comfort_hess_vv_n;
    double* comfort_hess_vphi_n;
    double* comfort_hess_phiphi_n;
    double* comfort_hess_aa_n;
};
#endif //COMFORT_TRAJECTORY_OPTIMIZER_ROBOTMODEL_H
