import pandas as pd
from typing import List, Dict
import covasim as cv

start_day = pd.to_datetime('2020-02-24').date()
lockdown_start = (pd.to_datetime('2020-03-08').date() - start_day).days
lockdown_end = (pd.to_datetime('2020-05-18').date() - start_day).days
summer_end = (pd.to_datetime('2020-09-15').date() - start_day).days
orange_zone_1 = (pd.to_datetime('2020-11-15').date() - start_day).days
yellow_zone_1 = (pd.to_datetime('2020-12-10').date() - start_day).days
orange_zone_2 = (pd.to_datetime('2020-12-21').date() - start_day).days
yellow_zone_2 = (pd.to_datetime('2021-02-01').date() - start_day).days
red_zone = (pd.to_datetime('2021-02-27').date() - start_day).days


def get_interventions(pars: Dict[str, float], detailed: bool = True) -> List[cv.Intervention]:
    interventions = []

    # QUAR_POLICY --> test both at the beginning and at the end of the quarantine
    # SENSITIVITY --> 80% (20% of false negatives: https://www.health.harvard.edu/blog/which-test-is-best-for-covid-19-2020081020734)
    # ILI_PREV --> prevalence of influenza-like-illness symptoms in the population
    # LOSS_PROB --> probability of the person being lost-to-follow-up (default 0%, i.e. no one lost to follow-up)
    # TEST_DELAY --> days for test result to be known (default 0, i.e. results available instantly)
    tests_intervention = cv.test_num(
        'new_tests',
        quar_policy='both',
        sensitivity=0.8,
        do_plot=False
    )
    interventions.append(tests_intervention)

    tracing_intervention = cv.contact_tracing(
        trace_probs=dict(h=1.0, s=pars['trace_prob'], w=pars['trace_prob'], c=0.0),
        trace_time=dict(h=0, s=pars['trace_time'], w=pars['trace_time']),
        do_plot=False
    )
    interventions.append(tracing_intervention)

    if not detailed:
        return interventions

    # schools closed from 08/03 to 15/09, then blended modality until orange zone
    school_closed_intervention = cv.clip_edges(
        days=[lockdown_start, summer_end, red_zone],
        changes=[0.0, pars['school_contacts'], 0.0],
        layers='s'
    )
    interventions.append(school_closed_intervention)

    # smart working from 08/03 to 18/05
    smart_working_intervention = cv.clip_edges(
        days=[lockdown_start, lockdown_end],
        changes=[pars['work_contacts'], 1.0],
        layers='w'
    )
    interventions.append(smart_working_intervention)

    # lockdown from 08/03, then no lockdown, then orange/yellow zones
    lockdown_intervention = cv.clip_edges(
        days=[lockdown_start, lockdown_end, orange_zone_1, yellow_zone_1, orange_zone_2, yellow_zone_2, red_zone],
        changes=[0.0, 1.0, pars['orange_contacts'], pars['yellow_contacts'], pars['orange_contacts'], pars['yellow_contacts'], 0.0],
        layers='c'
    )
    interventions.append(lockdown_intervention)

    # non-compulsory masks from may (but lower transmissibility due to summer), then compulsory masks from september
    masks_intervention = cv.change_beta(
        days=[lockdown_end, summer_end],
        changes=[pars['summer_masks'], pars['winter_masks']],
        layers=['s', 'w', 'c']
    )
    interventions.append(masks_intervention)

    # regional lockdowns to avoid imported cases
    imported_intervention = cv.dynamic_pars(
        n_imports=dict(
            days=[lockdown_end, orange_zone_1, yellow_zone_1, orange_zone_2, yellow_zone_2, red_zone],
            vals=[pars['summer_imports'], pars['orange_imports'], pars['yellow_imports'], pars['orange_imports'], pars['yellow_imports'], 0.0]
        )
    )
    interventions.append(imported_intervention)

    return interventions
