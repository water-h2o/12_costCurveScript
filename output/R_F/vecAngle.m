function angle = vecAngle(P_L, P_C, P_R)

    % each point is given as a 2D vector
    
    n_R = (P_R - P_C) / norm(P_R - P_C);
    n_L = (P_L - P_C) / norm(P_L - P_C);
    
    angle = rad2deg(atan2(norm(det([n_R; n_L])), dot(n_R, n_L)));
end