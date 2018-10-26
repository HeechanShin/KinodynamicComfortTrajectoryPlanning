//
// Created by heechanshin on 18. 9. 25.
//

#include <ros/ros.h>
#include <tf/tf.h>
#include <ackermann_msgs/AckermannDrive.h>
#include <nav_msgs/Odometry.h>
#include <bod_srvs/Bod.h>
#include <std_msgs/Float64MultiArray.h>

#include "ComfortOptimizer.h"

#define STATE_DIM 5
#define ACTION_DIM 2
#define N 100

int main(int argc, char **argv){
    Eigen::Matrix<double, (STATE_DIM+ACTION_DIM), 1> currOdom;
    Eigen::Matrix<double, (STATE_DIM+ACTION_DIM), 1> goalOdom;

    ros::init(argc, argv, "comfort_trajectory_optimizer");
    ros::NodeHandle n;

    ros::ServiceClient client = n.serviceClient<bod_srvs::Bod>("bod");
    ros::Publisher pubTrajectory = n.advertise<std_msgs::Float64MultiArray>("trajectory", 1);
    ros::Publisher pubDetectedObstacle = n.advertise<std_msgs::Float64MultiArray>("detected_obstacle", 1);

    while(ros::ok()){
        nav_msgs::Odometry::ConstPtr curr_odom = ros::topic::waitForMessage<nav_msgs::Odometry>("curr_odom", n);
        currOdom.setZero();
        currOdom(0, 0) = curr_odom->pose.pose.position.x;
        currOdom(1, 0) = curr_odom->pose.pose.position.y;
        tf::Pose pose_start;
        tf::poseMsgToTF(curr_odom->pose.pose, pose_start);
        currOdom(3, 0) = tf::getYaw(pose_start.getRotation()); // -pi ~ pi

        nav_msgs::Odometry::ConstPtr goal_odom = ros::topic::waitForMessage<nav_msgs::Odometry>("goal_odom", n);
        goalOdom.setZero();
        goalOdom(0, 0) = goal_odom->pose.pose.position.x;
        goalOdom(1, 0) = goal_odom->pose.pose.position.y;
        tf::Pose pose_goal;
        tf::poseMsgToTF(goal_odom->pose.pose, pose_goal);
        goalOdom(3, 0) = tf::getYaw(pose_goal.getRotation());

        auto robot = new RobotModel<STATE_DIM, ACTION_DIM>(N, client, pubTrajectory, pubDetectedObstacle);

        auto optimizer = new ComfortOptimizer<STATE_DIM, ACTION_DIM>(robot);
        SmartPtr<TNLP> tnlp = optimizer;

        SmartPtr<IpoptApplication> app = new IpoptApplication();
        ApplicationReturnStatus status;
        app->Options()->SetStringValue("linear_solver", "ma86");
        app->Options()->SetIntegerValue("max_iter", 300);
        app->Options()->SetNumericValue("tol", 1e-5);
        status = app->Initialize();

        if (status != Solve_Succeeded) {
            printf("\n\n*** Error during initialization!\n");
        }

        optimizer->plan(currOdom, goalOdom);
        status = app->OptimizeTNLP(tnlp);

        if (status == Solve_Succeeded) {
            // Retrieve some statistics about the solve
            int iter_count = app->Statistics()->IterationCount();
            printf("\n\n*** The problem solved in %d iterations!\n", iter_count);

            Number final_obj = app->Statistics()->FinalObjective();
            printf("\n\n*** The final value of the objective function is %e.\n", final_obj);

            double* finalSolution = robot->finalSolution;
        }
    }

    return 0;
}