import pandas as pd
from typing import List, Dict
import covasim as cv

start_day = pd.to_datetime('2020-02-24')
lockdown_start = (pd.to_datetime('2020-03-08') - start_day).days
lockdown_end = (pd.to_datetime('2020-05-18') - start_day).days
summer_end = (pd.to_datetime('2020-09-15') - start_day).days
orange_zone_1 = (pd.to_datetime('2020-11-15') - start_day).days
yellow_zone_1 = (pd.to_datetime('2020-12-10') - start_day).days
orange_zone_2 = (pd.to_datetime('2020-12-21') - start_day).days
yellow_zone_2 = (pd.to_datetime('2021-02-01') - start_day).days
red_zone = (pd.to_datetime('2021-02-27') - start_day).days


# SENSITIVITY: https://www.health.harvard.edu/blog/which-test-is-best-for-covid-19-2020081020734
def tests() -> cv.Intervention:
    return cv.test_num('new_tests', quar_policy='both', sensitivity=0.8, do_plot=False)

def contact_tracing(trace_prob: float, trace_time: float) -> cv.Intervention:
    return cv.contact_tracing(
        trace_probs=dict(h=1.0, s=trace_prob, w=trace_prob, c=0.0),
        trace_time=dict(h=0, s=trace_time, w=trace_time),
        do_plot=False
    )

# smart working from 08/03 to 18/05
def smart_working(work_contacts: float) -> cv.Intervention:
    return cv.clip_edges(days=[lockdown_start, lockdown_end], changes=[work_contacts, 1.0], layers='w')

# schools closed from 08/03 to 15/09, then blended modality until orange zone
def schools_closed(school_contacts: float) -> cv.Intervention:
    return cv.clip_edges(days=[lockdown_start, summer_end, red_zone], changes=[0.0, school_contacts, 0.0], layers='s')

# lockdown from 08/03, then no lockdown, then orange/yellow zones
def zones_lockdown(yellow_contacts: float, orange_contacts: float) -> cv.Intervention:
    return cv.clip_edges(
        days=[lockdown_start, lockdown_end, orange_zone_1, yellow_zone_1, orange_zone_2, yellow_zone_2, red_zone],
        changes=[0.0, 1.0, orange_contacts, yellow_contacts, orange_contacts, yellow_contacts, 0.0],
        layers='c'
    )

# non-compulsory masks from may (but lower transmissibility due to summer), then compulsory masks from september
def masks(summer_masks: float, winter_masks: float) -> cv.Intervention:
    return cv.change_beta(days=[lockdown_end, summer_end], changes=[summer_masks, winter_masks], layers=['s', 'w', 'c'])

# regional lockdowns to avoid imported cases
def imported_cases(summer_imp: float, yellow_imp: float, orange_imp: float) -> cv.Intervention:
    return cv.dynamic_pars(
        n_imports=dict(
            days=[lockdown_end, orange_zone_1, yellow_zone_1, orange_zone_2, yellow_zone_2, red_zone],
            vals=[summer_imp, orange_imp, yellow_imp, orange_imp, yellow_imp, 0.0]
        )
    )

# summer viral load reduction
def viral_load_reduction(
        initial_beta: float, summer_beta: float, winter_beta: float,
        initial_symp: float, summer_symp: float, winter_symp: float,
        initial_sev: float, summer_sev: float, winter_sev: float,
        initial_crit: float, summer_crit: float, winter_crit: float,
        initial_death: float, summer_death: float, winter_death: float,
) -> cv.Intervention:
    days = [0, lockdown_end, summer_end]
    dynamic_pars = {}
    if initial_beta is not None and summer_beta is not None and winter_beta is not None:
        dynamic_pars['beta'] = dict(days=days, vals=[initial_beta, summer_beta, winter_beta])
    if initial_symp is not None and summer_symp is not None and winter_symp is not None:
        dynamic_pars['rel_symp_prob'] = dict(days=days, vals=[initial_symp, summer_symp, winter_symp])
    if initial_sev is not None and summer_sev is not None and winter_sev is not None:
        dynamic_pars['rel_severe_prob'] = dict(days=days, vals=[initial_sev, summer_sev, winter_sev])
    if initial_crit is not None and summer_crit is not None and winter_crit is not None:
        dynamic_pars['rel_crit_prob'] = dict(days=days, vals=[initial_crit, summer_crit, winter_crit])
    if initial_death is not None and summer_death is not None and winter_death is not None:
        dynamic_pars['rel_death_prob'] = dict(days=days, vals=[initial_death, summer_death, winter_death])
    return cv.dynamic_pars(dynamic_pars)

def get_interventions(p: Dict[str, float]) -> List[cv.Intervention]:
    interventions = [tests()]

    if p.get('trace_prob') is not None and p.get('trace_time') is not None:
        interventions.append(contact_tracing(p['trace_prob'], p['trace_prob']))

    if p.get('work_contacts') is not None:
        interventions.append(smart_working(p['work_contacts']))

    if p.get('school_contacts') is not None:
        interventions.append(schools_closed(p['school_contacts']))

    if p.get('yellow_contacts') is not None and p.get('orange_contacts') is not None:
        interventions.append(zones_lockdown(p['yellow_contacts'], p['orange_contacts']))

    if p.get('summer_masks') is not None and p.get('winter_masks') is not None:
        interventions.append(masks(p['summer_masks'], p['winter_masks']))

    if p.get('summer_imp') is not None and p.get('yellow_imp') is not None and p.get('orange_imp') is not None:
        interventions.append(imported_cases(p['summer_imp'], p['yellow_imp'], p['orange_imp']))

    interventions.append(viral_load_reduction(
        p.get('initial_beta'), p.get('summer_beta'), p.get('winter_beta'),
        p.get('initial_symp'), p.get('summer_symp'), p.get('winter_symp'),
        p.get('initial_sev'), p.get('summer_sev'), p.get('winter_sev'),
        p.get('initial_crit'), p.get('summer_crit'), p.get('winter_crit'),
        p.get('initial_death'), p.get('summer_death'), p.get('winter_death'),
    ))

    return interventions
