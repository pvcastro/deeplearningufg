Highcharts.chart('container', {
    chart: {
        type: 'line'
    },
    title: {
        text: ''
    },
    xAxis: {
        categories: ['2009', '2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017']
    },
    yAxis: {
        title: {
            text: 'Milhões'
        }
    },
    plotOptions: {
        line: {
            dataLabels: {
                enabled: true
            },
            enableMouseTracking: false
        },
        series: {
        		label: {
                enabled: false
            }
        }
    },
    series: [{
    		name: "Casos pendentes",
        	data: [49.4,50.3,52.3,54.1,58.0,57.3,61.9,63.3,63.5]
    }, {
    		name: "Processos baixados",
        	data: [18.3,17.1,18.1,19.1,19.4,19.9,20.0,20.8,21.7]
    }, {
    		name: "Casos novos",
        	data: [17.8,17.5,18.6,19.8,20.5,20.3,19.5,19.8,20.2]
    }]
});