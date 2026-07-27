function [muU, sigmaU, muYimg, sigmaYimg] = loadStats(statsPath)
    S = load(statsPath);
    muU = S.mu_u;
    sigmaU = S.sigma_u;
    muYimg = S.mu_y;
    sigmaYimg = S.sigma_y;
end
