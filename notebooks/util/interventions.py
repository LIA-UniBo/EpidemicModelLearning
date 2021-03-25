import pandas as pd
from typing import List, Dict
import covasim as cv


class Interventions:
    def __init__(self, start_date='2020-02-24', test_column='new_tests', **kwargs: Dict[str, float]):
        self.start_date = start_date
        self.test_column = test_column
        self.parameters = kwargs

    def get_delta(self, date: str) -> int:
        return (pd.to_datetime(date).date() - pd.to_datetime(self.start_date).date()).days

    def get_values(self, postfix: str = '', **kwargs: Dict[str, float]) -> pd.Series:
        periods = pd.Series({
            self.start_date: 'init',
            '2020-03-08': 'red',
            '2020-05-18': 'summer',
            '2020-11-08': 'yellow',
            '2020-11-15': 'orange',
            '2020-12-10': 'yellow',
            '2020-12-21': 'orange',
            '2021-02-01': 'yellow',
            '2021-02-21': 'orange',
            '2021-03-01': 'red'
        }, name='periods')
        periods.index = [self.get_delta(d) for d in periods.index]
        return periods.map({k: kwargs.get(f'{k}{postfix}') for k in periods.unique()})

    # SENSITIVITY: https://www.health.harvard.edu/blog/which-test-is-best-for-covid-19-2020081020734
    def tests(self) -> cv.Intervention:
        return cv.test_num(self.test_column, quar_policy='both', sensitivity=0.8, do_plot=False)

    def contact_tracing(self) -> cv.Intervention:
        defaults = dict(
            household_trace_prob=1.0,
            household_trace_time=0.0,
            school_trace_prob=self.parameters.get('trace_prob', 0.0),
            school_trace_time=self.parameters.get('trace_time', 0.0),
            work_trace_prob=self.parameters.get('trace_prob', 0.0),
            work_trace_time=self.parameters.get('trace_time', 0.0),
            casual_trace_prob=0.0,
            casual_trace_time=0.0
        )
        defaults.update(self.parameters)
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
            ),
            do_plot=False
        )

    def smart_working(self) -> cv.Intervention:
        defaults = dict(
            init_work_contacts=1.,
            summer_work_contacts=1.,
            yellow_work_contacts=1.,
            orange_work_contacts=1.,
            red_work_contacts=self.parameters.get('work_contacts')
        )
        defaults.update(self.parameters)
        assert defaults['red_work_contacts'] is not None, 'red_work_contacts is required'
        v = self.get_values(postfix='_work_contacts', **defaults)
        return cv.clip_edges(days=v.index.values, changes=v.values, layers='w')

    def schools_closed(self) -> cv.Intervention:
        defaults = dict(
            init_school_contacts=1.,
            summer_school_contacts=0.,
            yellow_school_contacts=self.parameters.get('school_contacts'),
            orange_school_contacts=self.parameters.get('school_contacts'),
            red_school_contacts=0.
        )
        defaults.update(self.parameters)
        assert defaults['yellow_school_contacts'] is not None, 'yellow_school_contacts is required'
        assert defaults['orange_school_contacts'] is not None, 'orange_school_contacts is required'
        v = self.get_values(postfix='_school_contacts', **defaults)
        return cv.clip_edges(days=v.index.values, changes=v.values, layers='s')

    def lockdown_interactions(self) -> cv.Intervention:
        defaults = dict(
            init_casual_contacts=1.,
            summer_casual_contacts=1.,
            yellow_casual_contacts=1.,
            orange_casual_contacts=self.parameters.get('casual_contacts'),
            red_casual_contacts=0.
        )
        defaults.update(self.parameters)
        assert defaults['orange_casual_contacts'] is not None, 'orange_casual_contacts is required'
        v = self.get_values(postfix='_casual_contacts', **defaults)
        return cv.clip_edges(days=v.index.values, changes=v.values, layers='c')

    # regional lockdowns to avoid imported cases
    def imported_cases(self) -> cv.Intervention:
        defaults = dict(
            init_imports=0.,
            summer_imports=self.parameters.get('init_imports', 0.),
            yellow_imports=self.parameters.get('init_imports', 0.),
            orange_imports=self.parameters.get('init_imports', 0.),
            red_imports=self.parameters.get('init_imports', 0.)
        )
        defaults.update(self.parameters)
        v = self.get_values(postfix='_imports', **defaults)
        return cv.dynamic_pars(n_imports=dict(days=v.index.values, vals=v.values))

    # summer viral load reduction
    def viral_load_reduction(self) -> cv.Intervention:
        assert 'init_beta' in self.parameters
        days = [0, self.get_delta('2020-05-15'), self.get_delta('2020-09-15')]
        defaults = dict(
            init_beta=self.parameters['init_beta'],
            init_symp=self.parameters.get('init_symp', 1.),
            init_sev=self.parameters.get('init_sev', 1.),
            init_crit=self.parameters.get('init_crit', 1.),
            init_death=self.parameters.get('init_death', 1.)
        )
        args = {
            **defaults,
            **{key.replace('init', 'summer'): val for key, val in defaults.items()},
            **{key.replace('init', 'winter'): val for key, val in defaults.items()}
        }
        args.update(self.parameters)
        return cv.dynamic_pars(dict(
            beta=dict(days=days, vals=[args['init_beta'], args['summer_beta'], args['winter_beta']]),
            rel_symp_prob=dict(days=days, vals=[args['init_symp'], args['summer_symp'], args['winter_symp']]),
            rel_severe_prob=dict(days=days, vals=[args['init_sev'], args['summer_sev'], args['winter_sev']]),
            rel_crit_prob=dict(days=days, vals=[args['init_crit'], args['summer_crit'], args['winter_crit']]),
            rel_death_prob=dict(days=days, vals=[args['init_death'], args['summer_death'], args['winter_death']])
        ))

    def get_all(self, exclude=None) -> List[cv.Intervention]:
        exclude = [] if exclude is None else exclude
        interventions = [
            self.tests,
            self.contact_tracing,
            self.smart_working,
            self.schools_closed,
            self.lockdown_interactions,
            self.imported_cases,
            self.viral_load_reduction
        ]
        return [inter() for inter in interventions if inter.__name__ not in exclude]
