Highcharts.chart('container', {
    chart: {
        type: 'bar'
    },
    title: {
        text: ''
    },
    xAxis: {
        categories: ['TRT4', 'TRT15', 'TRT3', 'TRT1', 'TRT2', 'TRT9', 'TRT7', 'TRT18', 'TRT6', 'TRT5', 'TRT12', 'TRT10', 'TRT8', 'TRT11', 'TRT21', 'TRT22', 'TRT13', 'TRT14', 'TRT23', 'TRT16', 'TRT20', 'TRT19', 'TRT24', 'TRT17'],
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
        data: [99.2, 96.7, 96.2, 96.1, 91.9, 100.0, 99.1, 99.0, 98.4, 96.4, 95.1, 93.0, 87.9, 99.9, 99.5, 99.4, 99.4, 99.3, 98.8, 98.8, 98.8, 98.7, 98.1, 96.4]
    }]
});