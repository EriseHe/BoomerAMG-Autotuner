function spe10_run(root, count)
  global PILOT_DIR PILOT_CASE;
  PILOT_DIR=fullfile(root,'results/application_default_100/spe10');
  PILOT_CASE=0;
  run(fullfile(root,'results/application_default_100/MRST/startup.m'));
  addpath(fullfile(root,'experiments/application_pilot'));
  mrstDataDirectory(fullfile(root,'results/application_default_100/datasets'));
  mrstModule add spe10 incomp;
  gravity reset off; mrstVerbose off;
  [G,~,rock]=getSPE10setup(1:15);
  rock.poro=max(rock.poro,0.01);
  T=computeTrans(G,rock); pv=poreVolume(G,rock);
  horizon=10*year; dt=horizon/100;
  W=[];
  W=verticalWell(W,G,rock,1,1,[],'type','rate','val',sum(pv)/horizon,'comp_i',[1,0]);
  W=verticalWell(W,G,rock,60,220,[],'type','bhp','val',100*barsa,'comp_i',[1,0]);
  fluid=initSimpleFluid('mu',[1,5]*centi*poise,'n',[2,2],'rho',[0,0]);
  state=initResSol(G,0,[0,1]);
  fprintf('SPE10 cells %d; pressure systems %d\n',G.cells.num,count); fflush(stdout);
  for k=1:count
    ts=tic;
    state=incompTPFA(state,G,T,fluid,'wells',W,'reduce',true,'LinSolve',@spe10_case);
    pressure_wall=toc(ts);
    ts=tic;
    state=implicitTransport(state,G,dt,rock,fluid,'wells',W,'verbose',false);
    transport_wall=toc(ts);
    row=[k, pressure_wall,transport_wall,min(state.s(:,1)),max(state.s(:,1)),sum(pv.*state.s(:,1))];
    f=fopen(fullfile(PILOT_DIR,'application.csv'),'a');
    fprintf(f,'%.17g,%.17g,%.17g,%.17g,%.17g,%.17g\n',row);fclose(f);
    fprintf('SPE10 step %d pressure %.2fs transport %.2fs saturation [%.4g, %.4g]\n',k,pressure_wall,transport_wall,row(4),row(5));fflush(stdout);
  end
  save('-mat7-binary',fullfile(PILOT_DIR,'final_state.mat'),'state');
end
