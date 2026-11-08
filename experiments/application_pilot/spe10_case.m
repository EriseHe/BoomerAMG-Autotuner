function x = spe10_case(A, b)
  global PILOT_DIR PILOT_CASE;
  PILOT_CASE = PILOT_CASE + 1;
  % Export CSR through the CSC traversal of A transpose, preserving A exactly.
  [j,i,v] = find(A');
  n = size(A,1);
  ptr = [0; cumsum(accumarray(i,1,[n,1]))];
  f = fopen(fullfile(PILOT_DIR,'request.bin'),'wb');
  fwrite(f,[n;numel(v)],'int32');
  fwrite(f,ptr,'int32'); fwrite(f,j-1,'int32');
  fwrite(f,v,'double'); fwrite(f,b,'double'); fclose(f);
  f=fopen(fullfile(PILOT_DIR,'ready'),'w'); fprintf(f,'%d',PILOT_CASE);fclose(f);
  while ~exist(fullfile(PILOT_DIR,'response.ready'),'file'), pause(0.02); end
  f=fopen(fullfile(PILOT_DIR,'response.bin'),'rb');x=fread(f,n,'double');fclose(f);
  delete(fullfile(PILOT_DIR,'response.ready'));
  assert(numel(x)==n && all(isfinite(x)));
end
