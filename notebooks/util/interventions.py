import pandas as pd
from typing import List, Dict
import covasim as cv
from numba.tests.test_parallel_backend import mask_runner


def get_delta(date: str, start_date: str = '2020-02-24') -> int:
    return (pd.to_datetime(date).date() - pd.to_datetime(start_date).date()).days


def get_vals(default: float = None, white: float = None, yellow: float = None, orange: float = None, red: float = None):
    periods = pd.Series({
        '2020-03-08': 'red',
        '2020-05-18': 'white',
        '2020-11-08': 'yellow',
        '2020-11-15': 'orange',
        '2020-12-10': 'yellow',
        '2020-12-21': 'orange',
        '2021-02-01': 'yellow',
        '2021-02-21': 'orange',
        '2021-03-01': 'red'
    })
    periods.index = [get_delta(d) for d in periods.index]
    return periods if default is None else periods.map({
        'white': default if white is None else white,
        'yellow': default if yellow is None else yellow,
        'orange': default if orange is None else orange,
        'red': default if red is None else red
    })


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
    v = get_vals(default=1., red=work_contacts)
    return cv.clip_edges(days=v.index.values, changes=v.values, layers='w')


# schools closed from 08/03 to 15/09, then blended modality until orange zone
def schools_closed(school_contacts: float) -> cv.Intervention:
    v = get_vals(default=0., yellow=school_contacts, orange=school_contacts)
    return cv.clip_edges(days=v.index.values, changes=v.values, layers='s')


# lockdown from 08/03, then no lockdown, then orange/yellow zones
def lockdown_interactions(yellow_contacts: float, orange_contacts: float) -> cv.Intervention:
    v = get_vals(default=1., yellow=yellow_contacts, orange=orange_contacts, red=0.)
    return cv.clip_edges(days=v.index.values, changes=v.values, layers='c')


def masks(winter_change: float) -> cv.Intervention:
    return cv.change_beta(days=[get_delta('2020-09-15')], changes=[winter_change], layers=['w', 's', 'c'])


# regional lockdowns to avoid imported cases
def imported_cases(white_imp: float, yellow_imp: float, orange_imp: float) -> cv.Intervention:
    v = get_vals(default=0., white=white_imp, yellow=yellow_imp, orange=orange_imp)
    return cv.dynamic_pars(n_imports=dict(days=v.index.values, vals=v.values))


# summer viral load reduction
def viral_load_reduction(
        summer_beta: float, summer_symp: float, summer_sev: float, summer_crit: float, summer_death: float,
        winter_beta: float, winter_symp: float, winter_sev: float, winter_crit: float, winter_death: float
) -> cv.Intervention:
    days = [get_delta('2020-05-15'), get_delta('2020-09-15')]
    dynamic_pars = {}
    if summer_beta is not None and winter_beta is not None:
        dynamic_pars['beta'] = dict(days=days, vals=[summer_beta, winter_beta])
    if summer_symp is not None and winter_symp is not None:
        dynamic_pars['rel_symp_prob'] = dict(days=days, vals=[summer_symp, winter_symp])
    if summer_sev is not None and winter_sev is not None:
        dynamic_pars['rel_severe_prob'] = dict(days=days, vals=[summer_sev, winter_sev])
    if summer_crit is not None and winter_crit is not None:
        dynamic_pars['rel_crit_prob'] = dict(days=days, vals=[summer_crit, winter_crit])
    if summer_death is not None and winter_death is not None:
        dynamic_pars['rel_death_prob'] = dict(days=days, vals=[summer_death, winter_death])
    return None if len(dynamic_pars) == 0 else cv.dynamic_pars(dynamic_pars)


def get_interventions(p: Dict[str, float]) -> List[cv.Intervention]:
    interventions = [tests()]

    if p.get('trace_prob') is not None and p.get('trace_time') is not None:
        interventions.append(contact_tracing(p['trace_prob'], p['trace_time']))

    if p.get('work_contacts') is not None:
        interventions.append(smart_working(p['work_contacts']))

    if p.get('school_contacts') is not None:
        interventions.append(schools_closed(p['school_contacts']))

    if p.get('yellow_contacts') is not None and p.get('orange_contacts') is not None:
        interventions.append(lockdown_interactions(p['yellow_contacts'], p['orange_contacts']))

    if p.get('winter_beta_change') is not None:
        interventions.append(masks(p['winter_beta_change']))

    if p.get('white_imp') is not None and p.get('yellow_imp') is not None and p.get('orange_imp') is not None:
        interventions.append(imported_cases(p['white_imp'], p['yellow_imp'], p['orange_imp']))

    vlr = viral_load_reduction(
        p.get('summer_beta'), p.get('summer_symp'), p.get('summer_sev'), p.get('summer_crit'), p.get('summer_death'),
        p.get('winter_beta'), p.get('winter_symp'), p.get('winter_sev'), p.get('winter_crit'), p.get('winter_death')
    )
    if vlr is not None:
        interventions.append(vlr)

    return interventions
