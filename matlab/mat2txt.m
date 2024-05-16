%%
clear; close all;

%%
opts.ds = 10;

%%
mat = load('/home/manu/tmp/20240515-1315.mat');
sig = mat.A;

plot(sig(1:opts.ds:end));

% dlmwrite('/home/manu/tmp/data.txt', mat.A, 'delimiter', '\n', 'precision', 4);

%%