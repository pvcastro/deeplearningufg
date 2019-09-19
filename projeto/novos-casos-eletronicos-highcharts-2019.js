Highcharts.chart('container', {
    chart: {
        type: 'line'
    },
    title: {
        text: ''
    },
    xAxis: {
        categories: ['2009', '2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018']
    },
    yAxis: {
        title: {
            text: 'Percentual de casos novos eletrônicos (%)'
        },
        labels: {
        		formatter: function() {
               return Math.abs(this.value) + '%';
            }
        },
        max: 100
    },
    plotOptions: {
        line: {
            dataLabels: {
                enabled: true,
                formatter: function() {
                   return this.y + '%';
                }
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
    		name: "Estadual",
        	data: [4.2,5.7,10.9,13.8,22.3,35.3,49.7,69.9,78.0,82.6]
    }, {
    		name: "Trabalho",
        	data: [2.8,2.1,5.4,13.4,33.3,56.9,77.1,92.1,96.3,97.7]
    }]
});