locals {
  json_data = jsondecode(file("../config.json"))
}

provider "grafana" {
  url  = join("",["http://",local.json_data.master_ip, ":30003/"])
  auth = "admin:admin"
}

resource "grafana_data_source" "prometheusdb" {
  type          = "prometheus"
  name          = "Prometheus"
  url           = join("",["http://",local.json_data.master_ip, ":31113/"])
}

resource "grafana_data_source" "influxdb" {
  type          = "influxdb"
  name          = "myk6db"
  url           = "http://138.246.234.122:8086/"
  database_name      = "myk6db"
}

resource "grafana_dashboard" "Dashboard1" {
  config_json = file("faas_rev4.json")
}

resource "grafana_dashboard" "Dashboard2" {
  config_json = file("faas_rev2.json")
}

resource "grafana_dashboard" "Kube_Dashboard" {
  config_json = file("pod-metrics_rev2.json")
}

resource "grafana_dashboard" "K6_Dashboard" {
  config_json = file("k6-load-testing-results_rev3.json")
}
