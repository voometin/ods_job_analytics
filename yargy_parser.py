# -*- coding: utf-8 -*-


import re

from yargy import rule, and_, or_
from yargy.interpretation import (
    fact,
    const,
    attribute
)
from yargy.predicates import (
    eq, length_eq,
    in_, in_caseless,
    type,
    normalized, caseless, dictionary
)

Part = fact('Part', ['part'])

Money = fact(
    'Money',
    ['integer_min', attribute('integer_max', -1), attribute('currency', '-'), attribute('multiplier', -1), attribute('period', '-')]
)


DOT = eq('.')
INT = type('INT')

########
#
#   CURRENCY
#
##########


EURO = or_(
    normalized('евро'),
    normalized('euro'),
    eq('€'),
    caseless('EUR')
).interpretation(
    const('EUR')
)

DOLLARS = or_(
    normalized('доллар'),
    normalized('дол'),
    normalized('dollar'),
    eq('$'),
    caseless('USD')
).interpretation(
    const('USD')
)

RUBLES = or_(
    rule(normalized('ruble')),
    rule(normalized('рубль')),
    rule(normalized('рубл')),
    rule(
        or_(
            caseless('руб'),
            caseless('rub'),
#             caseless('rur'),
            caseless('р'),
            eq('₽')
        ),
        DOT.optional()
    )
).interpretation(
    const("RUB")
)

CURRENCY = or_(
    EURO,
    DOLLARS,
    RUBLES
).interpretation(
    Money.currency
)

############
#
#  MULTIPLIER
#
##########

MILLION = or_(
    rule(caseless('млн'), DOT.optional()),
    rule(normalized('миллион')),
    rule(in_('мМmM'))
).interpretation(
    const(10 ** 6)
)

THOUSAND = or_(
    rule(caseless('т'), DOT),
    rule(caseless('к')),
    rule(caseless('k')),
    rule(caseless('тыс'), DOT.optional()),
    rule(normalized('тысяча'))
).interpretation(
    const(10 ** 3)
)

MULTIPLIER = or_(
    MILLION,
    THOUSAND
).interpretation(
    Money.multiplier
)


#######
#
#   AMOUNT
#
########


def normalize_integer(value):
#     integer = re.sub('[\s\.,]+', '', value)
    integer = re.sub('[\s\.,](\S|$|\s$)', '\g<1>', value)
    return integer


PART = and_(
    INT,
    length_eq(3)
)

SEP = in_(',.')

INTEGER = or_(
    rule(INT),
    rule(INT, PART),
    rule(INT, PART, PART),
    rule(INT, SEP, PART),
    rule(INT, SEP, PART, SEP, PART),
)

# *Вилка*: 150к-250к (примерно 50-100, 120-180, 180-250 Junior/Middle/Senior) gross + премия 20% годового дохода по KPI
# give 150 -250, 50 - 100120, 180 - 180250 due to ',' - is in SEP for '5,000' and yargy ignore spaces
# and if we would ignore ',' in SEP we wouldn't able to catch '5,000' like samples


AMOUNT = rule(
    INTEGER,
#     rule(
#         SEP,
#         FRACTION
#     ).optional(),
    MULTIPLIER.optional(),
    # NUMERAL.optional()
)




# PERIODS = {
#     'день': dsl.DAY,
#     'сутки': dsl.DAY,
#     'час': dsl.HOUR,
#     'смена': dsl.SHIFT
# }
PERIODS = {
    'h', 'hour', 'ч', 'час', 'hr',
    
    'd', 'day', 'д', 'день', 'сутки', 'смена',
    
    'm',
    'month',
    'м'
    'месяц',
    'мес',
    
    'y',
    'year',
    'г',
    'год',
}

PERIOD = dictionary(
    PERIODS
)

PER = or_(
    eq('/'),
    in_caseless({'в', 'за', 'per'})
)

RATE = rule(
    PER,
    PERIOD.interpretation(Money.period)
)


MONEY = rule(
    or_(
        in_({'•', ':', '`', '~', '*', '-', '–', '—', ';', '.', '(', 'от', 'from'}),
        type('RU'),
        type('LATIN'),
       ).optional(),
    CURRENCY.interpretation(Money.currency).optional(),
    eq('+').optional(),
    eq('*').optional(),
    INTEGER.interpretation(Money.integer_min.custom(normalize_integer)),
    MULTIPLIER.interpretation(Money.multiplier).optional(),
    eq('*').optional(),
    CURRENCY.interpretation(Money.currency).optional(),
    eq('*').optional(),
    in_({'-', '–', '—', '–', ';', '.', 'до', 'to', 'и'}).optional(),
    eq('*').optional(),
    CURRENCY.interpretation(Money.currency).optional(),
    eq('*').optional(),
    INTEGER.interpretation(Money.integer_max.custom(normalize_integer)).optional(),
    eq('+').optional(),
    eq('*').optional(),
    in_({'.', ','}).optional(), 
    CURRENCY.interpretation(Money.currency).optional(),
    eq('*').optional(),
    MULTIPLIER.interpretation(Money.multiplier).optional(),
    eq('*').optional(),
    CURRENCY.interpretation(Money.currency).optional(),
    eq('*').optional(),
    RATE.optional(),
    eq('*').optional()
).interpretation(Money)

