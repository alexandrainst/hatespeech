'''Unit tests related to the data module.'''

import pytest
import pandas as pd
from src.data import load_data


class TestData:

    @pytest.fixture(scope='class')
    def data(self):
        return load_data()

    def test_data_is_dataframe(self, data):
        assert isinstance(data, pd.DataFrame)

    def test_data_columns(self, data):
        cols = ['account', 'text', 'date', 'action']
        assert data.columns.tolist() == cols

    def test_data_dtypes(self, data):
        dtypes = ['category', 'object', 'datetime64[ns]', 'category']
        assert data.dtypes.map(str).tolist() ==  dtypes

    def test_approximate_length(self, data):
        assert len(data) > 4_000_000 and len(data) < 5_000_000

    def test_deleted_hidden_length(self, data):
        deleted_hidden_data = len(data.query('action != "none"'))
        assert deleted_hidden_data > 50_000 and deleted_hidden_data < 100_000

    def test_accounts(self, data):
        accounts = {
            'aftenshowet - dr',
            'bag danmark med tobias - dr1',
            'bamse',
            'bedrag - dr',
            'bonderøven',
            'bruno bjørn',
            'deadline - dr',
            'debatten - dr',
            'detektor - dr',
            'dr',
            'dr big band',
            'dr e-sport',
            'dr ekstremsport',
            'dr historie',
            'dr koncerthuset',
            'dr kultur',
            'dr lyd',
            'dr mad',
            'dr minisjang & ramasjang',
            'dr nyheder',
            'dr p1',
            'dr p2',
            'dr p3',
            'dr p4 bornholm',
            'dr p4 fyn',
            'dr p4 københavn',
            'dr p4 midt & vest',
            'dr p4 nordjylland',
            'dr p4 sjælland',
            'dr p4 syd og esbjerg',
            'dr p4 trekanten',
            'dr p4 østjylland',
            'dr p5',
            'dr pigekoret',
            'dr politik',
            'dr ramasjang',
            'dr sporten',
            'dr symfoniorkestret',
            'dr ultra',
            'dr vejr',
            'dr videnskab',
            'dr1',
            'dr2',
            'dr2 dokumania',
            'dr3',
            'drtv',
            'fonk - det er lørdag på dr p4',
            'frank & kastaniegaarden',
            'herrens veje - dr',
            'karrierekanonen',
            'kontant - dr1',
            'lågsus - dr p3',
            'mgp - dr',
            'musikquizzen - dr p4',
            'nak & æd',
            'p3 - dr',
            'p3 x drtv',
            'p4 - dr',
            'p6 beat - dr',
            'p8 jazz - dr',
            'rigtige mænd - dr',
            'rytteriet',
            'sara & monopolet',
            'sort søndag',
            'spise med price - dr',
            'troldspejlet',
            'tæt på sandheden med jonatan spang',
            'vi ses hos clement'
        }
        assert set(data.account.unique().tolist()) == accounts
