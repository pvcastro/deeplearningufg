Highcharts.chart('container', {
    chart: {
        type: 'bar'
    },
    title: {
        text: ''
    },
    xAxis: {
        categories: ['TJSP', 'TJRJ', 'TJPR', 'TJRS', 'TJMG', 'TJSC', 'TJBA', 'TJMT', 'TJGO', 'TJPE', 'TJMA', 'TJCE', 'TJDFT', 'TJPA', 'TJES', 'TJTO', 'TJMS', 'TJAL', 'TJAM', 'TJAC', 'TJAP', 'TJSE', 'TJRO', 'TJPB', 'TJRR', 'TJRN', 'TJPI'],
        title: {
            text: null
        }
    },
    yAxis: {
        min: 0,
        title: {
            text: '',
            align: 'high'
        },
        labels: {
            overflow: 'justify'
        }
    },
    tooltip: {
        valueSuffix: ' %'
    },
    plotOptions: {
        bar: {
            dataLabels: {
                enabled: true
            }
        }
    },
    credits: {
        enabled: false
    },
    series: [{
        name: 'Percentual de casos novos eletrônicos',
        data: [95.8, 94.4, 92.4, 36.7, 31.0, 95.1, 90.3, 76.5, 74.1, 70.6, 55.9, 54.0, 53.0, 36.3, 26.1, 100.0, 100.0, 100.0, 100.0, 99.4, 96.8, 87.3, 84.3, 82.8, 73.0, 61.1, 54.4]
    }]
});