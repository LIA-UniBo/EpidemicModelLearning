import pandas as pd
import covasim as cv
from typing import List, Dict

DEFAULT_START = '2020-02-24'


def get_delta(date: str, start_date: str = DEFAULT_START) -> int:
    return (pd.to_datetime(date).date() - pd.to_datetime(start_date).date()).days


def get_values(periods: Dict[int, str], mappings: Dict[str, float], postfix: str = '') -> pd.Series:
    periods = pd.Series(periods)
    return periods.map({k: mappings.get(f'{k}{postfix}') for k in periods.unique()})


def tests(daily_test: object) -> cv.Intervention:
    return cv.test_num(daily_tests=daily_test, quar_policy='both', sensitivity=0.8)


def contact_tracing(parameters: Dict[str, float]) -> cv.Intervention:
    defaults = dict(
        household_trace_prob=1.0,
        household_trace_time=0.0,
        school_trace_prob=parameters.get('trace_prob', 0.0),
        school_trace_time=parameters.get('trace_time', 0.0),
        work_trace_prob=parameters.get('trace_prob', 0.0),
        work_trace_time=parameters.get('trace_time', 0.0),
        casual_trace_prob=0.0,
        casual_trace_time=0.0
    )
    defaults.update(parameters)
    return cv.contact_tracing(
        trace_probs=dict(
            h=defaults['household_trace_prob'],
            s=defaults['school_trace_prob'],
            w=defaults['work_trace_prob'],
            c=defaults['casual_trace_prob']
        ),
        trace_time=dict(
            h=defaults['household_trace_time'],
            s=defaults['school_trace_time'],
            w=defaults['work_trace_time'],
            c=defaults['casual_trace_time']
        )
    )


def smart_working(periods: pd.Series, parameters: Dict[str, float]) -> cv.Intervention:
    defaults = dict(
        init_work_contacts=1.,
        summer_work_contacts=1.,
        yellow_work_contacts=1.,
        orange_work_contacts=1.,
        red_work_contacts=parameters.get('work_contacts')
    )
    defaults.update(parameters)
    assert defaults['red_work_contacts'] is not None, 'red_work_contacts is required'
    v = get_values(periods, defaults, postfix='_work_contacts')
    return cv.clip_edges(days=v.index.values, changes=v.values, layers='w')


def schools_closed(periods: pd.Series, parameters: Dict[str, float]) -> cv.Intervention:
    defaults = dict(
        init_school_contacts=1.,
        summer_school_contacts=0.,
        yellow_school_contacts=parameters.get('school_contacts'),
        orange_school_contacts=parameters.get('school_contacts'),
        red_school_contacts=0.
    )
    defaults.update(parameters)
    assert defaults['yellow_school_contacts'] is not None, 'yellow_school_contacts is required'
    assert defaults['orange_school_contacts'] is not None, 'orange_school_contacts is required'
    v = get_values(periods, defaults, postfix='_school_contacts')
    return cv.clip_edges(days=v.index.values, changes=v.values, layers='s')


def lockdown_interactions(periods: pd.Series, parameters: Dict[str, float]) -> cv.Intervention:
    defaults = dict(
        init_casual_contacts=1.,
        summer_casual_contacts=1.,
        yellow_casual_contacts=parameters.get('casual_contacts'),
        orange_casual_contacts=parameters.get('casual_contacts'),
        red_casual_contacts=0.
    )
    defaults.update(parameters)
    assert defaults['yellow_casual_contacts'] is not None, 'yellow_casual_contacts is required'
    assert defaults['orange_casual_contacts'] is not None, 'orange_casual_contacts is required'
    v = get_values(periods, defaults, postfix='_casual_contacts')
    return cv.clip_edges(days=v.index.values, changes=v.values, layers='c')


# regional lockdowns to avoid imported cases
def imported_cases(periods: pd.Series, parameters: Dict[str, float]) -> cv.Intervention:
    defaults = dict(
        init_imports=parameters.get('init_imports', 0.),
        summer_imports=parameters.get('init_imports', 0.),
        yellow_imports=parameters.get('init_imports', 0.),
        orange_imports=parameters.get('init_imports', 0.),
        red_imports=0.
    )
    defaults.update(parameters)
    v = get_values(periods, defaults, postfix='_imports')
    return cv.dynamic_pars(n_imports=dict(days=v.index.values, vals=v.values))


# summer viral load reduction
def viral_load_reduction(parameters: Dict[str, float]) -> cv.Intervention:
    assert 'init_beta' in parameters
    days = [0, get_delta('2020-05-18'), get_delta('2020-10-01')]
    defaults = dict(
        init_beta=parameters['init_beta'],
        init_symp=parameters.get('init_symp', 1.),
        init_sev=parameters.get('init_sev', 1.),
        init_crit=parameters.get('init_crit', 1.),
        init_death=parameters.get('init_death', 1.)
    )
    args = {
        **defaults,
        **{key.replace('init', 'summer'): val for key, val in defaults.items()},
        **{key.replace('init', 'winter'): val for key, val in defaults.items()}
    }
    args.update(parameters)
    return cv.dynamic_pars(dict(
        beta=dict(days=days, vals=[args['init_beta'], args['summer_beta'], args['winter_beta']]),
        rel_symp_prob=dict(days=days, vals=[args['init_symp'], args['summer_symp'], args['winter_symp']]),
        rel_severe_prob=dict(days=days, vals=[args['init_sev'], args['summer_sev'], args['winter_sev']]),
        rel_crit_prob=dict(days=days, vals=[args['init_crit'], args['summer_crit'], args['winter_crit']]),
        rel_death_prob=dict(days=days, vals=[args['init_death'], args['summer_death'], args['winter_death']])
    ))


def get_interventions(periods: Dict[int, str],
                      parameters: Dict[str, float],
                      daily_tests: object = 'new_tests') -> List[cv.Intervention]:
    periods = pd.Series(periods)
    return [
        tests(daily_tests),
        contact_tracing(parameters),
        smart_working(periods, parameters),
        schools_closed(periods, parameters),
        lockdown_interactions(periods, parameters),
        imported_cases(periods, parameters),
        viral_load_reduction(parameters)
    ]


def get_calibration_interventions(parameters: Dict[str, float]):
    return get_interventions(
        periods={get_delta(d): z for d, z in {
            DEFAULT_START: 'init',
            '2020-03-08': 'red',
            '2020-05-18': 'summer',
            '2020-11-08': 'yellow',
            '2020-11-15': 'orange',
            '2020-12-10': 'yellow',
            '2020-12-21': 'orange',
            '2021-02-01': 'yellow',
            '2021-02-21': 'orange',
            '2021-03-01': 'red'
        }.items()},
        parameters=parameters
    )


def get_sampling_interventions(zones: List[str], parameters: Dict[str, float], interval: int = 15):
    mapping = dict(W='init', Y='yellow', O='orange', R='red')
    return get_interventions(
        periods={
            0: 'init',
            get_delta('2020-03-08'): 'red',
            get_delta('2020-05-18'): 'summer',
            **{get_delta(f'2020-11-01') + interval * i: mapping[z] for i, z in enumerate(zones)}
        },
        parameters=parameters
    )
