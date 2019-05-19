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
        }
    },
    series: [{
    		name: "Casos pendentes",
        	data: [3.3,3.4,3.4,4.0,4.5,4.6,5.1,5.4,5.5]
    }, {
    		name: "Processos baixados",
        	data: [3.3,3.4,3.7,3.8,4.0,4.2,4.3,4.2,4.5]
    }, {
    		name: "Casos novos",
        	data: [3.4,3.3,3.7,3.9,4.0,4.0,4.1,4.3,4.3]
    }]
});