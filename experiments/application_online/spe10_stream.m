function spe10_stream(root, output, count, steps_per_scenario)
  global PILOT_DIR PILOT_CASE;
  PILOT_DIR=output; PILOT_CASE=0;
  run(fullfile(root,'results/application_default_100/MRST/startup.m'));
  addpath(fullfile(root,'experiments/application_pilot'));
  mrstDataDirectory(fullfile(root,'results/application_default_100/datasets'));
  mrstModule add spe10 incomp;
  gravity reset off; mrstVerbose off;
  [G,~,rock]=getSPE10setup(1:15); rock.poro=max(rock.poro,0.01);
  T=computeTrans(G,rock); pv=poreVolume(G,rock);
  horizon=10*year; dt=horizon/100; W=[];
  W=verticalWell(W,G,rock,1,1,[],'type','rate','val',sum(pv)/horizon,'comp_i',[1,0]);
  W=verticalWell(W,G,rock,60,220,[],'type','bhp','val',100*barsa,'comp_i',[1,0]);
  campaign=jsondecode(fileread(fullfile(output,'campaign.json')));
  xl=G.cells.centroids(:,1)<mean(G.cells.centroids(:,1));
  yl=G.cells.centroids(:,2)<mean(G.cells.centroids(:,2));
  for k=1:count
    scenario=floor((k-1)/steps_per_scenario)+1;
    local_step=mod(k-1,steps_per_scenario)+1;
    settings=campaign(scenario);
    if local_step==1
      state=initResSol(G,0,[0,1]); injected=0;
      fluid=initSimpleFluid('mu',[1,settings.viscosity_ratio]*centi*poise,'n',[2,2],'rho',[0,0]);
    end
    rate=1;
    if strcmp(settings.schedule,'pulsed')
      rates=[0.5,1.5,0.5,1.5]; rate=rates(floor((local_step-1)/25)+1);
    end
    W(1).val=rate*sum(pv)/horizon;
    ts=tic; s=state.s(:,1);
    context=[1,log(settings.viscosity_ratio)/log(20),mean(s),std(s,1),mean(s(xl)),mean(s(~xl)),mean(s(yl)),mean(s(~yl)),injected,rate];
    context_sec=toc(ts);
    f=fopen(fullfile(output,'context.json'),'w');
    fprintf(f,'%s',jsonencode(struct('context',context,'context_sec',context_sec,'physical_time_years',(local_step-1)*.1,'scenario',scenario,'scenario_step',local_step)));fclose(f);
    ts=tic;
    state=incompTPFA(state,G,T,fluid,'wells',W,'reduce',true,'LinSolve',@spe10_case);
    pressure_wall=toc(ts); ts=tic;
    state=implicitTransport(state,G,dt,rock,fluid,'wells',W,'verbose',false);
    transport_wall=toc(ts);
    injected=injected+rate/100;
    f=fopen(fullfile(output,'application.csv'),'a');
    fprintf(f,'%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%d,%d\n',k,pressure_wall,transport_wall,min(state.s(:,1)),max(state.s(:,1)),sum(pv.*state.s(:,1)),scenario,local_step);fclose(f);
    if local_step==steps_per_scenario
      save('-mat7-binary',fullfile(output,sprintf('final_state_%02d.mat',scenario)),'state');
    end
  end
  save('-mat7-binary',fullfile(output,'final_state.mat'),'state');
end
