# para o latex
atl.to_ltx(P_t, frmt = "{:6.4f}", arraytype = "table",row = False)

tab_95 = Stargazer([reg_t_95,reg_r_95])
tab_99 = Stargazer([reg_t_99,reg_r_99])
open('rho95.tex', 'w').write(tab_95.render_latex())
open('rho99.tex', 'w').write(tab_99.render_latex())

