from math import sqrt, sin, cos, atan, atan2, degrees, pi
import numpy as np
from scripts.utils import EndEffector, rotm_to_euler, euler_to_rotm, check_joint_limits, dh_to_matrix, near_zero

NUM_DOF = 5
L1 = 0.155
L2 = 0.099
L3 = 0.095
L4 = 0.055
L5 = 0.105

def calc_numerical_ik(desired: list, thetas, tol=0.01, ilimit=50):
        """ Calculate numerical inverse kinematics based on input coordinates. """
        
        
        pos_des = [desired[0], desired[1], desired[2]]
        print(f"running numerical inverse for {pos_des}")

        # theta to update and check against
        new_thetas = thetas
        for _ in range(ilimit):
            # solve for error
            pos_current = solve_forward_kinematics(new_thetas, radians=False)
            error = pos_des - pos_current[0:3]
            # If error outside tol, recalculate theta (Newton-Raphson)
            if np.linalg.norm(error) > tol:
                new_thetas = new_thetas + np.dot(inverse_jacobian(thetas=new_thetas, pseudo=True), error)
            # If error is within tolerence: break
            else:
                break


        return new_thetas

def solve_forward_kinematics(theta: list, radians=True):
        """Solves the new transformation matrix from base to EE at current theta values"""

        # Convert degrees to radians
        if not radians:
            for i in range(len(theta)):
                theta[i] = np.deg2rad(theta[i])

        # DH parameters = [theta, d, a, alpha]
        DH = np.zeros((5, 4))
        DH[0] = [theta[0],          L1,     0,  np.pi/2]
        DH[1] = [theta[1]+np.pi/2,  0,      L2, np.pi]
        DH[2] = [theta[2],          0,      L3, np.pi]
        DH[3] = [theta[3]-np.pi/2,  0,      0,  -np.pi/2]
        DH[4] = [theta[4],          L4+L5,  0,  0]

        T = np.zeros((NUM_DOF,4,4))
        for i in range(NUM_DOF):
            T[i] = dh_to_matrix(DH[i])

        return T[0] @ T[1] @ T[2] @ T[3] @ T[4] @ np.array([0, 0, 0, 1])

def jacobian(theta: list = None):
        """
        Compute the Jacobian matrix for the current robot configuration.

        Args:
            theta (list, optional): The joint angles for the robot. Defaults to self.theta.
        Returns:
            Jacobian matrix (3x5).
        """

        # Define DH parameters
        DH = np.zeros((5, 4))
        DH[0] = [theta[0], L1, 0, np.pi/2]
        DH[1] = [theta[1] + np.pi/2, 0, L2, np.pi]
        DH[2] = [theta[2], 0, L3, np.pi]
        DH[3] = [theta[3] - np.pi/2, 0, 0, -np.pi/2]
        DH[4] = [theta[4], L4 + L5, 0, 0]

        # Compute transformation matrices
        T = np.zeros((NUM_DOF,4,4))
        for i in range(NUM_DOF):
            T[i] = dh_to_matrix(DH[i])

        # Precompute transformation matrices for efficiency
        T_cumulative = [np.eye(4)]
        for i in range(NUM_DOF):
            T_cumulative.append(T_cumulative[-1] @ T[i])

        # Define O0 for calculations
        O0 = np.array([0, 0, 0, 1])
        
        # Initialize the Jacobian matrix
        jacobian = np.zeros((3, NUM_DOF))

        # Calculate the Jacobian columns
        for i in range(NUM_DOF):
            T_curr = T_cumulative[i]
            T_final = T_cumulative[-1]
            
            # Calculate position vector r
            r = (T_final @ O0 - T_curr @ O0)[:3]

            # Compute the rotation axis z
            z = T_curr[:3, :3] @ np.array([0, 0, 1])

            # Compute linear velocity part of the Jacobian
            jacobian[:, i] = np.cross(z, r)

        # Replace near-zero values with zero, primarily for debugging purposes
        return near_zero(jacobian)

def inverse_jacobian(thetas: list, pseudo=False):
        """
        Compute the inverse of the Jacobian matrix using either pseudo-inverse or regular inverse.

        Args:
            pseudo: Boolean flag to use pseudo-inverse (default is False).

        Returns:
            The inverse (or pseudo-inverse) of the Jacobian matrix.
        """

        J = jacobian()
        JT = np.transpose(J)
        manipulability_idx = np.sqrt(np.linalg.det(J @ JT))

        if pseudo:
            return np.linalg.pinv(jacobian())
        else:
            return np.linalg.inv(jacobian())
