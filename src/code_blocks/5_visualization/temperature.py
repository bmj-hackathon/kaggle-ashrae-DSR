# from IPython.core.display import display, HTML
#
# # --- plotly ---
import plotly.io as pio
pio.renderers.default = "browser"
# pio.renderers.default = "png"
# pio.renderers.default = "firefox"
# import cufflinks as cf
from plotly.offline import download_plotlyjs, init_notebook_mode, iplot
# cf.go_offline()
# # from plotly import tools, subplots
# import plotly.offline as py
# # py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
# import plotly.express as px
# import plotly.figure_factory as ff
# import plotly.io as pio
# pio.renderers.default = "browser"
#%%
# t = np.linspace(0, 10, 50)
# x, y, z = np.cos(t), np.sin(t), t

bldg_id = df_train['building_id'].unique()[0]
this_bldg = util_data.get_building(df_train, bldg_id)
this_site_leak_df = leak_dfs[1].rename({'meter_reading_scraped':'meter_reading'}, axis=1)
this_bldg_leaked = util_data.get_building(this_site_leak_df, bldg_id)

fig = go.Figure()
fig.add_trace(go.Scatter(x=this_bldg.index, y=this_bldg['electricity']))
fig.add_trace(go.Scatter(x=this_bldg_leaked.index, y=this_bldg_leaked['electricity']))
fig.show()
